// rq-d4c2b6cd — Steepest-descent minimizer with adaptive step size.
// See `rqm/minimization/steepest-descent.md`.

use cudarc::driver::CudaSlice;
use serde::Deserialize;

use crate::forces::ForceField;
use crate::gpu::{GpuContext, GpuError, ParticleBuffers, compute_total_potential_energy};
use crate::integrator::Constraint;
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{
    Minimizer, MinimizerBuilder, MinimizerConvergence, MinimizerError, MinimizerStepReport,
    sd_compute_step, sd_f_max_reduction, sd_restore, sd_snapshot,
};

// rq-0a2ca9ac — `[minimization.algorithm]` schema fields
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SteepestDescentParams {
    #[serde(default = "default_initial_step")]
    pub initial_step: f64,
    #[serde(default = "default_max_step")]
    pub max_step: f64,
    #[serde(default = "default_step_increase")]
    pub step_increase: f64,
    #[serde(default = "default_step_decrease")]
    pub step_decrease: f64,
    #[serde(default = "default_force_tolerance")]
    pub force_tolerance: f64,
    #[serde(default = "default_energy_tolerance")]
    pub energy_tolerance: f64,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u64,
}

fn default_initial_step() -> f64 {
    1.0e-12
}
fn default_max_step() -> f64 {
    1.0e-10
}
fn default_step_increase() -> f64 {
    1.2
}
fn default_step_decrease() -> f64 {
    0.2
}
fn default_force_tolerance() -> f64 {
    1.0e-10
}
fn default_energy_tolerance() -> f64 {
    1.0e-7
}
fn default_max_iterations() -> u64 {
    1000
}

fn deserialize_params(params: &toml::Value) -> Result<SteepestDescentParams, ConfigError> {
    params
        .clone()
        .try_into::<SteepestDescentParams>()
        .map_err(|e| crate::io::config::translate_params_error("minimization.algorithm", e))
}

// rq-dc3b1bb5
#[derive(Debug)]
pub struct SteepestDescentMinimizer {
    initial_step: f32,
    max_step: f32,
    step_increase: f32,
    step_decrease: f32,
    force_tolerance: f64,
    energy_tolerance: f64,
    max_iterations: u64,

    current_step: f32,
    last_accepted_energy: f64,

    snapshot_x: CudaSlice<f32>,
    snapshot_y: CudaSlice<f32>,
    snapshot_z: CudaSlice<f32>,
    f_max_scratch: CudaSlice<f32>,
    pe_scratch: CudaSlice<f32>,
}

impl SteepestDescentMinimizer {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        params: &SteepestDescentParams,
    ) -> Result<Self, GpuError> {
        let snapshot_len = particle_count.max(1);
        let snapshot_x = gpu.device.alloc_zeros::<f32>(snapshot_len)?;
        let snapshot_y = gpu.device.alloc_zeros::<f32>(snapshot_len)?;
        let snapshot_z = gpu.device.alloc_zeros::<f32>(snapshot_len)?;
        let f_max_scratch = gpu.device.alloc_zeros::<f32>(1)?;
        let pe_scratch = gpu.device.alloc_zeros::<f32>(1)?;
        Ok(SteepestDescentMinimizer {
            initial_step: params.initial_step as f32,
            max_step: params.max_step as f32,
            step_increase: params.step_increase as f32,
            step_decrease: params.step_decrease as f32,
            force_tolerance: params.force_tolerance,
            energy_tolerance: params.energy_tolerance,
            max_iterations: params.max_iterations,
            current_step: params.initial_step as f32,
            last_accepted_energy: 0.0,
            snapshot_x,
            snapshot_y,
            snapshot_z,
            f_max_scratch,
            pe_scratch,
        })
    }
}

impl Minimizer for SteepestDescentMinimizer {
    // rq-39ab27d9 — Initial iteration: warm-up F_max and energy at accepted positions.
    fn initial_state(
        &mut self,
        buffers: &mut ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(f64, f64), MinimizerError> {
        let max_force = if buffers.particle_count() == 0 {
            0.0
        } else {
            timings.kernel_start(KernelStage::SD_F_MAX_REDUCTION)?;
            let f_max = sd_f_max_reduction(buffers, &mut self.f_max_scratch)?;
            timings.kernel_stop(KernelStage::SD_F_MAX_REDUCTION)?;
            f_max as f64
        };
        timings.kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)?;
        let energy = compute_total_potential_energy(buffers, &mut self.pe_scratch)? as f64;
        timings.kernel_stop(KernelStage::POTENTIAL_ENERGY_REDUCE)?;
        self.last_accepted_energy = energy;
        self.current_step = self.initial_step;
        Ok((energy, max_force))
    }

    // rq-3f080cf2 rq-cc9f4623 — Per-iteration sequence; empty-state branch.
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        force_field: &mut ForceField,
        constraint: Option<&mut dyn Constraint>,
        timings: &mut Timings,
    ) -> Result<MinimizerStepReport, MinimizerError> {
        let prev_energy = self.last_accepted_energy;
        let step_used = self.current_step;

        if buffers.particle_count() == 0 {
            // Empty system: no work to do; report a trivially accepted
            // zero-energy iteration so the runner's convergence check
            // can terminate.
            return Ok(MinimizerStepReport {
                accepted: true,
                energy: 0.0,
                max_force: 0.0,
                step_size: step_used as f64,
                prev_energy,
            });
        }

        // Compute F_max at the current (accepted) positions. The
        // runner is responsible for ensuring `forces_*` is current
        // before calling `step()`.
        timings.kernel_start(KernelStage::SD_F_MAX_REDUCTION)?;
        let f_max = sd_f_max_reduction(buffers, &mut self.f_max_scratch)?;
        timings.kernel_stop(KernelStage::SD_F_MAX_REDUCTION)?;

        if !(f_max > 0.0) {
            // Already at a force-zero configuration (or non-finite).
            // Report a "rejected" iteration that doesn't move; the
            // runner's convergence check picks this up via the
            // `max_force == 0` branch.
            return Ok(MinimizerStepReport {
                accepted: false,
                energy: prev_energy,
                max_force: 0.0,
                step_size: step_used as f64,
                prev_energy,
            });
        }

        // Snapshot positions for rejection rollback.
        timings.kernel_start(KernelStage::SD_SNAPSHOT)?;
        sd_snapshot(
            buffers,
            &mut self.snapshot_x,
            &mut self.snapshot_y,
            &mut self.snapshot_z,
        )?;
        timings.kernel_stop(KernelStage::SD_SNAPSHOT)?;

        // Apply the trial step: x_trial = x + step · F / F_max.
        let inv_f_max = 1.0_f32 / f_max;
        timings.kernel_start(KernelStage::SD_COMPUTE_STEP)?;
        sd_compute_step(buffers, step_used, inv_f_max)?;
        timings.kernel_stop(KernelStage::SD_COMPUTE_STEP)?;

        // Project constraints (if any) before evaluating trial energy.
        if let Some(c) = constraint {
            c.apply_position_projection_only(buffers, sim_box, timings)?;
        }

        // Evaluate forces and potential energy at the trial position.
        // Minimization needs energy every iteration to decide accept/reject,
        // so always request ForcesAndScalars.
        force_field.step(
            buffers,
            sim_box,
            timings,
            crate::forces::AggregateLevel::ForcesAndScalars,
        )?;
        timings.kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)?;
        let trial_energy =
            compute_total_potential_energy(buffers, &mut self.pe_scratch)? as f64;
        timings.kernel_stop(KernelStage::POTENTIAL_ENERGY_REDUCE)?;

        if trial_energy < prev_energy {
            // Accept. Compute F_max at the new (accepted) positions
            // for the convergence-criterion check.
            timings.kernel_start(KernelStage::SD_F_MAX_REDUCTION)?;
            let f_max_new = sd_f_max_reduction(buffers, &mut self.f_max_scratch)?;
            timings.kernel_stop(KernelStage::SD_F_MAX_REDUCTION)?;
            self.last_accepted_energy = trial_energy;
            self.current_step =
                (self.current_step * self.step_increase).min(self.max_step);
            Ok(MinimizerStepReport {
                accepted: true,
                energy: trial_energy,
                max_force: f_max_new as f64,
                step_size: step_used as f64,
                prev_energy,
            })
        } else {
            // Reject. Restore positions, re-evaluate forces and
            // potential energy at the restored positions so the next
            // iteration's `sd_compute_step` reads valid forces.
            timings.kernel_start(KernelStage::SD_RESTORE)?;
            sd_restore(
                buffers,
                &self.snapshot_x,
                &self.snapshot_y,
                &self.snapshot_z,
            )?;
            timings.kernel_stop(KernelStage::SD_RESTORE)?;
            force_field.step(
                buffers,
                sim_box,
                timings,
                crate::forces::AggregateLevel::ForcesAndScalars,
            )?;
            self.current_step = self.current_step * self.step_decrease;
            // F_max at the restored (accepted) positions.
            timings.kernel_start(KernelStage::SD_F_MAX_REDUCTION)?;
            let f_max_now = sd_f_max_reduction(buffers, &mut self.f_max_scratch)?;
            timings.kernel_stop(KernelStage::SD_F_MAX_REDUCTION)?;
            Ok(MinimizerStepReport {
                accepted: false,
                energy: prev_energy,
                max_force: f_max_now as f64,
                step_size: step_used as f64,
                prev_energy,
            })
        }
    }

    // rq-1440b6e6
    fn check_convergence(
        &self,
        report: &MinimizerStepReport,
    ) -> Option<MinimizerConvergence> {
        // Force-zero: no force to descend along; treat as converged.
        if report.max_force == 0.0 {
            return Some(MinimizerConvergence::ForceZero);
        }
        // Force tolerance: only applies when explicitly enabled
        // (force_tolerance > 0) and is checked at the accepted state.
        if self.force_tolerance > 0.0 && report.max_force <= self.force_tolerance {
            return Some(MinimizerConvergence::ForceTolerance);
        }
        // Energy tolerance: only triggers on an accepted iteration
        // (the relative change between two distinct accepted energies).
        if report.accepted
            && self.energy_tolerance > 0.0
            && (report.energy - report.prev_energy).abs()
                <= self.energy_tolerance
                    * report
                        .energy
                        .abs()
                        .max(report.prev_energy.abs())
                        .max(1.0e-30)
        {
            return Some(MinimizerConvergence::EnergyTolerance);
        }
        None
    }

    fn max_iterations(&self) -> u64 {
        self.max_iterations
    }
}

// rq-dddb8e7a — concrete MinimizerBuilder implementation for kind = "steepest-descent".
#[derive(Debug, Clone)]
pub struct SteepestDescentBuilder;

impl MinimizerBuilder for SteepestDescentBuilder {
    fn kind_name(&self) -> &'static str {
        "steepest-descent"
    }

    // rq-e2bb500b — schema cross-validation: domain checks on every algorithm field.
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive(
            "minimization.algorithm.initial_step",
            p.initial_step,
        )?;
        require_finite_positive(
            "minimization.algorithm.max_step",
            p.max_step,
        )?;
        if p.max_step < p.initial_step {
            return Err(ConfigError::InvalidValue {
                field: "minimization.algorithm.max_step".to_string(),
                reason: format!(
                    "max_step ({}) must be >= initial_step ({})",
                    p.max_step, p.initial_step
                ),
            });
        }
        if !p.step_increase.is_finite() || p.step_increase < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "minimization.algorithm.step_increase".to_string(),
                reason: format!(
                    "step_increase must be finite and >= 1.0, got {}",
                    p.step_increase
                ),
            });
        }
        if !p.step_decrease.is_finite()
            || !(p.step_decrease > 0.0 && p.step_decrease < 1.0)
        {
            return Err(ConfigError::InvalidValue {
                field: "minimization.algorithm.step_decrease".to_string(),
                reason: format!(
                    "step_decrease must be finite and in (0.0, 1.0), got {}",
                    p.step_decrease
                ),
            });
        }
        if !p.force_tolerance.is_finite() || p.force_tolerance < 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "minimization.algorithm.force_tolerance".to_string(),
                reason: format!(
                    "force_tolerance must be finite and >= 0.0, got {}",
                    p.force_tolerance
                ),
            });
        }
        if !p.energy_tolerance.is_finite() || p.energy_tolerance < 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "minimization.algorithm.energy_tolerance".to_string(),
                reason: format!(
                    "energy_tolerance must be finite and >= 0.0, got {}",
                    p.energy_tolerance
                ),
            });
        }
        if p.max_iterations == 0 {
            return Err(ConfigError::InvalidValue {
                field: "minimization.algorithm.max_iterations".to_string(),
                reason: "max_iterations must be strictly positive".to_string(),
            });
        }
        Ok(())
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        _n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Minimizer>, MinimizerError> {
        let p = deserialize_params(params).map_err(|_| {
            MinimizerError::UnknownKind("steepest-descent (malformed params)".into())
        })?;
        let m = SteepestDescentMinimizer::new(gpu, particle_count, &p)?;
        Ok(Box::new(m))
    }

    fn box_clone(&self) -> Box<dyn MinimizerBuilder> {
        Box::new(self.clone())
    }
}

fn require_finite_positive(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ConfigError::InvalidValue {
            field: field.to_string(),
            reason: format!("expected finite and strictly positive, got {value}"),
        });
    }
    Ok(())
}
