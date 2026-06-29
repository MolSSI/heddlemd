// rq-709c8eb5 — Analytical SETTLE constraint algorithm for symmetric
// three-atom rigid water (Miyamoto-Kollman 1992). See
// rqm/integration/settle.md.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use serde::Deserialize;

use crate::forces::{ConstraintList, GroupConstraint};
use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, settle_positions, settle_positions_no_velocity,
    settle_snapshot, settle_velocities, settle_virial_scatter,
};
use crate::io::config::{ConfigError, NamedSlotConfig};
use crate::kernels;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

use super::constraint::{Constraint, ConstraintBuilder, ConstraintError};
use crate::precision::Real;

/// SETTLE handles exactly one cluster shape: symmetric three-atom water.
pub const SETTLE_ATOMS: usize = 3;

// rq-55f60603 rq-eecd4961
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettleParams {
    /// Apex (oxygen)–hydrogen bond length, atomic units after config load.
    #[serde(rename = "d_OH")]
    pub d_oh: f64,
    /// Hydrogen–hydrogen distance, atomic units after config load.
    #[serde(rename = "d_HH")]
    pub d_hh: f64,
}

fn deserialize_params(params: &toml::Value) -> Result<SettleParams, ConfigError> {
    params
        .clone()
        .try_into::<SettleParams>()
        .map_err(|e| crate::io::config::translate_params_error("constraint_types", e))
}

fn validate_settle_params(name: &str, p: &SettleParams) -> Result<(), ConfigError> {
    if !p.d_oh.is_finite() || p.d_oh <= 0.0 {
        return Err(ConfigError::SettleParamsMalformed {
            name: name.to_string(),
            reason: format!("d_OH must be strictly positive and finite, got {}", p.d_oh),
        });
    }
    if !p.d_hh.is_finite() || p.d_hh <= 0.0 {
        return Err(ConfigError::SettleParamsMalformed {
            name: name.to_string(),
            reason: format!("d_HH must be strictly positive and finite, got {}", p.d_hh),
        });
    }
    // The apex height h = sqrt(d_OH² − (d_HH/2)²) must be real and
    // positive (the three atoms are not collinear).
    if p.d_hh >= 2.0 * p.d_oh {
        return Err(ConfigError::SettleParamsMalformed {
            name: name.to_string(),
            reason: format!(
                "d_HH must be less than 2 * d_OH (got d_HH = {}, d_OH = {}); the geometry would be collinear",
                p.d_hh, p.d_oh
            ),
        });
    }
    Ok(())
}

/// Canonical geometry of a SETTLE water group: distances of the atoms
/// from the centre of mass in the molecular frame. rq-57db9db2
#[derive(Debug, Clone, Copy)]
pub struct SettleGeometry {
    pub ra: Real,
    pub rb: Real,
    pub rc: Real,
}

/// Compute the canonical geometry `(ra, rb, rc)` from the bond distances
/// and masses. rq-57db9db2
fn canonical_geometry(d_oh: f64, d_hh: f64, m_o: f64, m_h: f64) -> SettleGeometry {
    let total = m_o + 2.0 * m_h;
    let h = (d_oh * d_oh - (d_hh * 0.5) * (d_hh * 0.5)).sqrt();
    SettleGeometry {
        ra: ((2.0 * m_h / total) * h) as Real,
        rb: ((m_o / total) * h) as Real,
        rc: (d_hh * 0.5) as Real,
    }
}

// rq-709c8eb5
#[derive(Debug, thiserror::Error)]
pub enum SettleError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("settle constraint type `{name}` is malformed: {reason}")]
    MalformedSettleType { name: String, reason: String },
    #[error("settle constraint group {group_index} has invalid shape: {reason}; use SHAKE for general rigid clusters")]
    InvalidGroupShape { group_index: usize, reason: String },
}

impl From<SettleError> for ConstraintError {
    fn from(e: SettleError) -> Self {
        match e {
            SettleError::Gpu(g) => ConstraintError::Gpu(g),
            SettleError::Timings(t) => ConstraintError::Timings(t),
            SettleError::MalformedSettleType { name, reason } => {
                ConstraintError::InvalidGroupShape {
                    group_index: 0,
                    kind: "settle".to_string(),
                    reason: format!("settle type {name}: {reason}"),
                }
            }
            SettleError::InvalidGroupShape { group_index, reason } => {
                ConstraintError::InvalidGroupShape {
                    group_index,
                    kind: "settle".to_string(),
                    reason,
                }
            }
        }
    }
}

// rq-709c8eb5
#[derive(Debug)]
pub struct SettleConstraintsState {
    pub device: Arc<CudaDevice>,
    pub group_count: usize,
    pub particle_count: usize,
    pub atom_slot_count: usize,
    pub group_atoms: CudaSlice<u32>,
    pub group_atom_offset: CudaSlice<u32>,
    pub group_atom_count: CudaSlice<u32>,
    pub group_ra: CudaSlice<Real>,
    pub group_rb: CudaSlice<Real>,
    pub group_rc: CudaSlice<Real>,
    pub group_m_o: CudaSlice<Real>,
    pub group_m_h: CudaSlice<Real>,
    pub snapshot_x: CudaSlice<Real>,
    pub snapshot_y: CudaSlice<Real>,
    pub snapshot_z: CudaSlice<Real>,
    pub constraint_virial: CudaSlice<Real>,
}

impl SettleConstraintsState {
    // rq-709c8eb5
    pub fn new(
        device: Arc<CudaDevice>,
        list: &ConstraintList,
        masses: &[Real],
        constraint_types: &[NamedSlotConfig],
    ) -> Result<Self, SettleError> {
        // Per-type validation: deserialise and bound-check every settle
        // type's params. Non-settle entries belong to other algorithms.
        for ct in constraint_types {
            if ct.kind != "settle" {
                continue;
            }
            let p = ct
                .params
                .clone()
                .try_into::<SettleParams>()
                .map_err(|e| SettleError::MalformedSettleType {
                    name: ct.name.clone(),
                    reason: e.to_string(),
                })?;
            if let Err(e) = validate_settle_params(&ct.name, &p) {
                let reason = match e {
                    ConfigError::SettleParamsMalformed { reason, .. } => reason,
                    other => other.to_string(),
                };
                return Err(SettleError::MalformedSettleType {
                    name: ct.name.clone(),
                    reason,
                });
            }
        }

        let n_groups = list.groups.len();

        let mut group_atoms_host: Vec<u32> = Vec::with_capacity(SETTLE_ATOMS * n_groups);
        let mut group_atom_offset_host: Vec<u32> = Vec::with_capacity(n_groups);
        let mut group_atom_count_host: Vec<u32> = Vec::with_capacity(n_groups);
        let mut group_ra_host: Vec<Real> = Vec::with_capacity(n_groups);
        let mut group_rb_host: Vec<Real> = Vec::with_capacity(n_groups);
        let mut group_rc_host: Vec<Real> = Vec::with_capacity(n_groups);
        let mut group_m_o_host: Vec<Real> = Vec::with_capacity(n_groups);
        let mut group_m_h_host: Vec<Real> = Vec::with_capacity(n_groups);

        for (gi, g) in list.groups.iter().enumerate() {
            let ct = &constraint_types[g.constraint_type_index as usize];
            let params: SettleParams = ct
                .params
                .clone()
                .try_into()
                .map_err(|e| SettleError::MalformedSettleType {
                    name: ct.name.clone(),
                    reason: e.to_string(),
                })?;

            if g.atom_count as usize != SETTLE_ATOMS {
                return Err(SettleError::InvalidGroupShape {
                    group_index: gi,
                    reason: format!(
                        "atom count {} (SETTLE requires exactly {})",
                        g.atom_count, SETTLE_ATOMS
                    ),
                });
            }

            let atoms = &list.group_atoms
                [g.atom_offset as usize..(g.atom_offset + g.atom_count) as usize];
            let m_o = masses[atoms[0] as usize];
            let m_h1 = masses[atoms[1] as usize];
            let m_h2 = masses[atoms[2] as usize];
            if m_h1 != m_h2 {
                return Err(SettleError::InvalidGroupShape {
                    group_index: gi,
                    reason: format!(
                        "the two hydrogens must have equal masses (got {m_h1} and {m_h2})"
                    ),
                });
            }

            let geom = canonical_geometry(params.d_oh, params.d_hh, m_o as f64, m_h1 as f64);

            let atom_offset = group_atoms_host.len() as u32;
            group_atoms_host.extend_from_slice(atoms);
            group_atom_offset_host.push(atom_offset);
            group_atom_count_host.push(g.atom_count);
            group_ra_host.push(geom.ra);
            group_rb_host.push(geom.rb);
            group_rc_host.push(geom.rc);
            group_m_o_host.push(m_o);
            group_m_h_host.push(m_h1);
        }

        let atom_slot_count = group_atoms_host.len();

        let group_atoms = device
            .htod_sync_copy(&pad_min1_u32(&group_atoms_host))
            .map_err(GpuError::from)?;
        let group_atom_offset = device
            .htod_sync_copy(&pad_min1_u32(&group_atom_offset_host))
            .map_err(GpuError::from)?;
        let group_atom_count = device
            .htod_sync_copy(&pad_min1_u32(&group_atom_count_host))
            .map_err(GpuError::from)?;
        let group_ra = device
            .htod_sync_copy(&pad_min1_real(&group_ra_host))
            .map_err(GpuError::from)?;
        let group_rb = device
            .htod_sync_copy(&pad_min1_real(&group_rb_host))
            .map_err(GpuError::from)?;
        let group_rc = device
            .htod_sync_copy(&pad_min1_real(&group_rc_host))
            .map_err(GpuError::from)?;
        let group_m_o = device
            .htod_sync_copy(&pad_min1_real(&group_m_o_host))
            .map_err(GpuError::from)?;
        let group_m_h = device
            .htod_sync_copy(&pad_min1_real(&group_m_h_host))
            .map_err(GpuError::from)?;
        let snapshot_x = device
            .alloc_zeros::<Real>(atom_slot_count.max(1))
            .map_err(GpuError::from)?;
        let snapshot_y = device
            .alloc_zeros::<Real>(atom_slot_count.max(1))
            .map_err(GpuError::from)?;
        let snapshot_z = device
            .alloc_zeros::<Real>(atom_slot_count.max(1))
            .map_err(GpuError::from)?;
        let constraint_virial = device
            .alloc_zeros::<Real>(atom_slot_count.max(1))
            .map_err(GpuError::from)?;

        Ok(SettleConstraintsState {
            device,
            group_count: n_groups,
            particle_count: list.particle_count,
            atom_slot_count,
            group_atoms,
            group_atom_offset,
            group_atom_count,
            group_ra,
            group_rb,
            group_rc,
            group_m_o,
            group_m_h,
            snapshot_x,
            snapshot_y,
            snapshot_z,
            constraint_virial,
        })
    }
}

fn pad_min1_u32(v: &[u32]) -> Vec<u32> {
    if v.is_empty() { vec![0u32] } else { v.to_vec() }
}
fn pad_min1_real(v: &[Real]) -> Vec<Real> {
    if v.is_empty() { vec![0.0] } else { v.to_vec() }
}

impl Constraint for SettleConstraintsState {
    // rq-709c8eb5 — snapshot the pre-drift positions; the position reset
    // uses them as the constraint-gradient reference frame.
    fn apply_before_drift(
        &mut self,
        buffers: &mut ParticleBuffers,
        _sim_box: &SimulationBox,
        _dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        if self.group_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SETTLE_SNAPSHOT)?;
        settle_snapshot(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &mut self.snapshot_x,
            &mut self.snapshot_y,
            &mut self.snapshot_z,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_SNAPSHOT)?;
        Ok(())
    }

    fn apply_after_drift(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        if self.group_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SETTLE_POSITIONS)?;
        settle_positions(
            buffers,
            &self.snapshot_x,
            &self.snapshot_y,
            &self.snapshot_z,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_ra,
            &self.group_rb,
            &self.group_rc,
            &self.group_m_o,
            &self.group_m_h,
            sim_box,
            dt,
            &mut self.constraint_virial,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_POSITIONS)?;
        Ok(())
    }

    fn apply_after_kick(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        if self.group_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SETTLE_VELOCITIES)?;
        settle_velocities(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_m_o,
            &self.group_m_h,
            sim_box,
            dt,
            &mut self.constraint_virial,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_VELOCITIES)?;
        timings.kernel_start(KernelStage::SETTLE_VIRIAL_SCATTER)?;
        settle_virial_scatter(
            buffers,
            &self.constraint_virial,
            &self.group_atoms,
            self.atom_slot_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_VIRIAL_SCATTER)?;
        Ok(())
    }

    fn apply_initial_velocity_projection(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        if self.group_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SETTLE_VELOCITIES)?;
        settle_velocities(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_m_o,
            &self.group_m_h,
            sim_box,
            0.0,
            &mut self.constraint_virial,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_VELOCITIES)?;
        Ok(())
    }

    fn apply_position_projection_only(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        if self.group_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SETTLE_POSITIONS_NO_VELOCITY)?;
        settle_positions_no_velocity(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_ra,
            &self.group_rb,
            &self.group_rc,
            &self.group_m_o,
            &self.group_m_h,
            sim_box,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_POSITIONS_NO_VELOCITY)?;
        Ok(())
    }

    fn group_count(&self) -> usize {
        self.group_count
    }
}

// rq-709c8eb5
#[derive(Debug, Clone)]
pub struct SettleBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for SettleBuilder {
    fn kind_name(&self) -> &'static str {
        "settle"
    }
}

impl ConstraintBuilder for SettleBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        validate_settle_params("<name placeholder>", &p)
    }

    fn expected_atom_count(&self, _params: &toml::Value) -> usize {
        SETTLE_ATOMS
    }

    // rq-709c8eb5 — synthesise the canonical water constraint pattern
    // from d_OH / d_HH so the framework's exclusion, virial, and DOF
    // machinery sees three constraints per group.
    fn expand_constraints(
        &self,
        params: &toml::Value,
    ) -> Result<Vec<GroupConstraint>, ConstraintError> {
        let p = deserialize_params(params).map_err(|e| ConstraintError::InvalidGroupShape {
            group_index: 0,
            kind: "settle".to_string(),
            reason: e.to_string(),
        })?;
        let d_oh = p.d_oh as Real;
        let d_hh = p.d_hh as Real;
        Ok(vec![
            GroupConstraint { local_i: 0, local_j: 1, r0: d_oh },
            GroupConstraint { local_i: 0, local_j: 2, r0: d_oh },
            GroupConstraint { local_i: 1, local_j: 2, r0: d_hh },
        ])
    }

    fn validate_group_shape(
        &self,
        group_index: usize,
        atoms: &[u32],
        _constraints: &[GroupConstraint],
        _params: &toml::Value,
        masses: &[Real],
    ) -> Result<(), ConstraintError> {
        if atoms.len() != SETTLE_ATOMS {
            return Err(ConstraintError::InvalidGroupShape {
                group_index,
                kind: "settle".to_string(),
                reason: format!(
                    "atom count {} (SETTLE requires exactly {})",
                    atoms.len(),
                    SETTLE_ATOMS
                ),
            });
        }
        let m_h1 = masses[atoms[1] as usize];
        let m_h2 = masses[atoms[2] as usize];
        if m_h1 != m_h2 {
            return Err(ConstraintError::InvalidGroupShape {
                group_index,
                kind: "settle".to_string(),
                reason: format!(
                    "the two hydrogens must have equal masses (got {m_h1} and {m_h2}); use SHAKE for asymmetric clusters"
                ),
            });
        }
        Ok(())
    }

    fn build(
        &self,
        device: Arc<CudaDevice>,
        _gpu: &GpuContext,
        _particle_count: usize,
        list: &ConstraintList,
        masses: &[Real],
        constraint_types: &[NamedSlotConfig],
    ) -> Result<Box<dyn Constraint>, ConstraintError> {
        let state = SettleConstraintsState::new(device, list, masses, constraint_types)?;
        Ok(Box::new(state))
    }
}

// rq-709c8eb5
#[derive(Debug, Clone)]
pub struct SettleKernels {
    pub settle_snapshot: CudaFunction,
    pub settle_positions: CudaFunction,
    pub settle_velocities: CudaFunction,
    pub settle_virial_scatter: CudaFunction,
    pub settle_positions_no_velocity: CudaFunction,
}

impl SettleKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::SETTLE),
            "settle",
            &[
                "settle_snapshot",
                "settle_positions",
                "settle_velocities",
                "settle_virial_scatter",
                "settle_positions_no_velocity",
            ],
        )?;
        Ok(SettleKernels {
            settle_snapshot: get_func(device, "settle", "settle_snapshot")?,
            settle_positions: get_func(device, "settle", "settle_positions")?,
            settle_velocities: get_func(device, "settle", "settle_velocities")?,
            settle_virial_scatter: get_func(device, "settle", "settle_virial_scatter")?,
            settle_positions_no_velocity: get_func(
                device,
                "settle",
                "settle_positions_no_velocity",
            )?,
        })
    }
}
