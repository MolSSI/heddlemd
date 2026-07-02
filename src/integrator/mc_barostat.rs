// rq-09ac44ea — Monte-Carlo barostat. Periodic Metropolis volume moves
// on molecular centres of mass. See `rqm/integration/mc-barostat.md`.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use serde::Deserialize;

use crate::forces::{AggregateLevel, ForceField, MoleculeList};
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_total_potential_energy,
    mc_barostat_scale_molecule_com,
};
use crate::integrator::philox::philox_4x32_10;
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::precision::{Real, Real4};
use crate::timings::{KernelStage, Timings};

use super::{Barostat, BarostatBuilder, BarostatError, BarostatPeriodicity, Constraint};
use crate::registry::KindedBuilder;

/// Number of attempted moves per adaptive-step adjustment window.
const ADAPT_INTERVAL: u32 = 10;

fn default_frequency() -> u32 {
    25
}

// rq-c6ee2fb9
#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct McBarostatParams {
    pub pressure: crate::units::Pressure,
    pub temperature: crate::units::Temperature,
    #[serde(default = "default_frequency")]
    pub frequency: u32,
    /// Initial maximum volume displacement in atomic units (`a_0^3`).
    /// When omitted, defaults to one percent of the initial box volume.
    #[serde(default)]
    pub volume_step: Option<f64>,
    pub seed: u64,
}

fn deserialize_params(params: &toml::Value) -> Result<McBarostatParams, ConfigError> {
    params
        .clone()
        .try_into::<McBarostatParams>()
        .map_err(|e| crate::io::config::translate_params_error("barostat", e))
}

fn require_finite(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(ConfigError::InvalidValue {
            field: field.to_string(),
            reason: format!("value must be finite, got {value}"),
        });
    }
    Ok(())
}

fn require_finite_positive(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ConfigError::InvalidValue {
            field: field.to_string(),
            reason: format!("value must be finite and strictly positive, got {value}"),
        });
    }
    Ok(())
}

// rq-09ac44ea
#[derive(Debug)]
pub struct McBarostat {
    device: Arc<CudaDevice>,
    pub pressure: f64,
    pub temperature: f64,
    pub frequency: u32,
    /// Configured initial volume step; `None` means "default to
    /// `0.01 · V_0`", resolved in `init_run`.
    volume_step_config: Option<f64>,
    pub max_volume_step: f64,
    pub seed: u64,
    pub draw_counter: u64,
    n_attempted: u32,
    n_accepted: u32,
    pub attempted_moves: u64,
    pub accepted_moves: u64,
    pub cumulative_barostat_injection: f64,
    pub most_recent_volume: f64,
    n_molecules: usize,
    mol_atom_offsets: CudaSlice<u32>,
    mol_atom_indices: CudaSlice<u32>,
    pe_scratch: CudaSlice<Real>,
    /// Pre-move device-resident snapshots for revert-on-reject
    /// (device-to-device copies; no host round-trip). Sized to the
    /// particle count in `init_run`.
    pos_snapshot: CudaSlice<Real4>,
    force_snapshot_x: CudaSlice<Real>,
    force_snapshot_y: CudaSlice<Real>,
    force_snapshot_z: CudaSlice<Real>,
}

impl McBarostat {
    fn new(
        gpu: &GpuContext,
        pressure: f64,
        temperature: f64,
        frequency: u32,
        volume_step: Option<f64>,
        seed: u64,
    ) -> Result<Self, GpuError> {
        let device = gpu.device.clone();
        let pe_scratch = device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        // Placeholder molecule tables; `init_run` uploads the real ones.
        let mol_atom_offsets = device.alloc_zeros::<u32>(1).map_err(GpuError::from)?;
        let mol_atom_indices = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;
        // Placeholder snapshot buffers; `init_run` sizes them to N.
        let pos_snapshot = device.alloc_zeros::<Real4>(0).map_err(GpuError::from)?;
        let force_snapshot_x = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let force_snapshot_y = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let force_snapshot_z = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        Ok(McBarostat {
            device,
            pressure,
            temperature,
            frequency,
            volume_step_config: volume_step,
            max_volume_step: volume_step.unwrap_or(0.0),
            seed,
            draw_counter: 0,
            n_attempted: 0,
            n_accepted: 0,
            attempted_moves: 0,
            accepted_moves: 0,
            cumulative_barostat_injection: 0.0,
            most_recent_volume: 0.0,
            n_molecules: 0,
            mol_atom_offsets,
            mol_atom_indices,
            pe_scratch,
            pos_snapshot,
            force_snapshot_x,
            force_snapshot_y,
            force_snapshot_z,
        })
    }

    fn acceptance_ratio(&self) -> f64 {
        if self.attempted_moves == 0 {
            0.0
        } else {
            self.accepted_moves as f64 / self.attempted_moves as f64
        }
    }

    /// Adaptive-step bookkeeping (algorithm step 10): retune
    /// `max_volume_step` toward ~50% acceptance every `ADAPT_INTERVAL`
    /// attempts.
    fn record_attempt(&mut self, accepted: bool, current_volume: f64) {
        self.attempted_moves += 1;
        self.n_attempted += 1;
        if accepted {
            self.accepted_moves += 1;
            self.n_accepted += 1;
        }
        if self.n_attempted >= ADAPT_INTERVAL {
            let attempted = self.n_attempted as f64;
            let accepted_n = self.n_accepted as f64;
            if accepted_n < 0.25 * attempted {
                self.max_volume_step /= 1.1;
            } else if accepted_n > 0.75 * attempted {
                self.max_volume_step = (self.max_volume_step * 1.1).min(0.3 * current_volume);
            }
            self.n_attempted = 0;
            self.n_accepted = 0;
        }
    }
}

impl Barostat for McBarostat {
    // rq-0ba1a24a
    fn periodicity(&self) -> BarostatPeriodicity {
        BarostatPeriodicity::EveryNSteps(self.frequency)
    }

    // rq-3e1fba8b — upload molecule tables, resolve the default volume
    // step, and seed the diagnostic volume.
    fn init_run(
        &mut self,
        sim_box: &SimulationBox,
        molecules: &MoleculeList,
    ) -> Result<(), BarostatError> {
        self.n_molecules = molecules.molecule_count;
        self.mol_atom_offsets = self
            .device
            .htod_sync_copy(&molecules.mol_atom_offsets)
            .map_err(GpuError::from)?;
        self.mol_atom_indices = if molecules.mol_atom_indices.is_empty() {
            self.device.alloc_zeros::<u32>(0).map_err(GpuError::from)?
        } else {
            self.device
                .htod_sync_copy(&molecules.mol_atom_indices)
                .map_err(GpuError::from)?
        };
        let n = molecules.particle_count;
        self.pos_snapshot = self.device.alloc_zeros::<Real4>(n).map_err(GpuError::from)?;
        self.force_snapshot_x = self.device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        self.force_snapshot_y = self.device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        self.force_snapshot_z = self.device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let v0 = sim_box.volume() as f64;
        self.max_volume_step = self.volume_step_config.unwrap_or(0.01 * v0);
        self.most_recent_volume = v0;
        Ok(())
    }

    // rq-03a5a290 rq-8114b8c4 — host-orchestrated Metropolis volume move.
    fn apply_move(
        &mut self,
        force_field: &mut ForceField,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        _constraint: Option<&mut dyn Constraint>,
        _dt: Real,
        timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        // 1. Current energy at the pre-move configuration. The runner
        //    evaluates scalars on the step immediately before a move
        //    boundary, so `buffers.potential_energies` already holds the
        //    per-particle potential energy of the current configuration —
        //    reduce it directly rather than re-running the force pipeline.
        timings.kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)?;
        let u_old = compute_total_potential_energy(buffers, &mut self.pe_scratch)? as f64;
        timings.kernel_stop(KernelStage::POTENTIAL_ENERGY_REDUCE)?;

        // 2. Snapshot box (host fields), positions and forces (device-to-
        //    device — no host round-trip). Flush the box so host fields
        //    are current. `buffers.forces_*` hold `F` at the current
        //    configuration (from the pre-move dynamics step); the
        //    snapshot restores them on a rejected move.
        sim_box.flush_from_device()?;
        let lattice_pre = sim_box.lattice();
        let v_old = sim_box.volume() as f64;
        let min_width_pre = sim_box.min_perpendicular_width() as f64;
        self.device
            .dtod_copy(&buffers.posq, &mut self.pos_snapshot)
            .map_err(GpuError::from)?;
        self.device
            .dtod_copy(&buffers.forces_x, &mut self.force_snapshot_x)
            .map_err(GpuError::from)?;
        self.device
            .dtod_copy(&buffers.forces_y, &mut self.force_snapshot_y)
            .map_err(GpuError::from)?;
        self.device
            .dtod_copy(&buffers.forces_z, &mut self.force_snapshot_z)
            .map_err(GpuError::from)?;

        // 3. Pre-increment the draw counter.
        self.draw_counter += 1;
        let draw_lo = self.draw_counter as u32;
        let draw_hi = (self.draw_counter >> 32) as u32;
        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let out = philox_4x32_10(seed_lo, seed_hi, draw_lo, draw_hi, 0, 0);
        let scale_u = 1.0_f64 / 4_294_967_296.0;
        let u1 = (out[0] as f64 + 0.5) * scale_u;
        let u2 = (out[1] as f64 + 0.5) * scale_u;

        // 4. Propose a volume change.
        let dv = self.max_volume_step * (2.0 * u1 - 1.0);
        let v_new = v_old + dv;
        let kt = self.temperature;

        // Width guard: an isotropic scale multiplies every perpendicular
        // width by `scale`. Reject a contraction that would drop the
        // minimum width below the neighbour search radius.
        let r_search = force_field
            .neighbor_list
            .as_ref()
            .and_then(|nl| nl.cell_list_data())
            .map(|cl| (cl.r_cut + cl.r_skin) as f64);

        let mut accepted = false;
        let mut v_post = v_old;

        // The minimum image is well-defined only when every interaction
        // radius is below half the shortest perpendicular width, i.e.
        // `min_width >= 2 · r_search`. An isotropic scale multiplies every
        // perpendicular width by `scale`.
        let early_reject = v_new <= 0.0 || {
            let scale = (v_new / v_old).cbrt();
            matches!(r_search, Some(rs) if scale * min_width_pre < 2.0 * rs)
        };

        if !early_reject {
            let scale = (v_new / v_old).cbrt();
            // 5/6. Apply the trial: scale box + molecular COMs.
            sim_box
                .multiply_lattice_isotropic(scale as Real)?;
            timings.kernel_start(KernelStage::MC_BAROSTAT_SCALE_COM)?;
            mc_barostat_scale_molecule_com(
                buffers,
                sim_box,
                &self.mol_atom_offsets,
                &self.mol_atom_indices,
                scale as Real,
            )?;
            timings.kernel_stop(KernelStage::MC_BAROSTAT_SCALE_COM)?;

            // 7. Trial energy at the scaled configuration.
            force_field.step(buffers, sim_box, timings, AggregateLevel::ForcesAndScalars)?;
            timings.kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)?;
            let u_new = compute_total_potential_energy(buffers, &mut self.pe_scratch)? as f64;
            timings.kernel_stop(KernelStage::POTENTIAL_ENERGY_REDUCE)?;

            // 8. Metropolis test for the COM-scaling NPT volume move.
            let w = (u_new - u_old) + self.pressure * dv
                - (self.n_molecules as f64) * kt * (v_new / v_old).ln();
            accepted = w <= 0.0 || u2 < (-w / kt).exp();

            if accepted {
                v_post = v_new;
            } else {
                // 9. Revert positions, forces, and the box (device-to-
                //    device restore from the snapshots).
                self.device
                    .dtod_copy(&self.pos_snapshot, &mut buffers.posq)
                    .map_err(GpuError::from)?;
                self.device
                    .dtod_copy(&self.force_snapshot_x, &mut buffers.forces_x)
                    .map_err(GpuError::from)?;
                self.device
                    .dtod_copy(&self.force_snapshot_y, &mut buffers.forces_y)
                    .map_err(GpuError::from)?;
                self.device
                    .dtod_copy(&self.force_snapshot_z, &mut buffers.forces_z)
                    .map_err(GpuError::from)?;
                sim_box
                    .set_lattice(
                        lattice_pre[0],
                        lattice_pre[1],
                        lattice_pre[2],
                        lattice_pre[3],
                        lattice_pre[4],
                        lattice_pre[5],
                    )?;
            }
        }

        // 10/11. Adaptive bookkeeping + diagnostics.
        self.record_attempt(accepted, v_post);
        self.most_recent_volume = v_post;
        if accepted {
            self.cumulative_barostat_injection += self.pressure * (v_post - v_old);
        }
        Ok(())
    }

    // rq-d0506951
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        use crate::units::Dimension;
        &[
            ("box_volume", Dimension::Dimensionless),
            ("mc_acceptance", Dimension::Dimensionless),
            ("mc_volume_step", Dimension::Dimensionless),
            ("mc_conserved", Dimension::Energy),
        ]
    }

    fn log_column_values(&self, kinetic_energy: f64, potential_energy: f64) -> Vec<f64> {
        let conserved = kinetic_energy
            + potential_energy
            + self.pressure * self.most_recent_volume
            - self.cumulative_barostat_injection;
        vec![
            self.most_recent_volume,
            self.acceptance_ratio(),
            self.max_volume_step,
            conserved,
        ]
    }
}

// rq-6e1916c0
#[derive(Debug, Clone)]
pub struct McBarostatBuilder;

impl KindedBuilder for McBarostatBuilder {
    fn kind_name(&self) -> &'static str {
        "monte-carlo"
    }
    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), crate::io::config::ConfigError> {
        crate::registry::convert_params_in_place::<McBarostatParams>(units, params)
    }
}

impl BarostatBuilder for McBarostatBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite("barostat.pressure", p.pressure.0)?;
        require_finite_positive("barostat.temperature", p.temperature.0)?;
        if p.frequency == 0 {
            return Err(ConfigError::InvalidValue {
                field: "barostat.frequency".to_string(),
                reason: "value must be >= 1, got 0".to_string(),
            });
        }
        if let Some(vs) = p.volume_step {
            require_finite_positive("barostat.volume_step", vs)?;
        }
        Ok(())
    }

    fn build(
        &self,
        gpu: &GpuContext,
        _particle_count: usize,
        _n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Barostat>, BarostatError> {
        let p = deserialize_params(params)
            .map_err(|_| BarostatError::UnknownKind("monte-carlo (malformed params)".into()))?;
        let state = McBarostat::new(
            gpu,
            p.pressure.0,
            p.temperature.0,
            p.frequency,
            p.volume_step,
            p.seed,
        )?;
        Ok(Box::new(state))
    }
}
