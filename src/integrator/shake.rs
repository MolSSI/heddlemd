// rq-9a80c43c — General SHAKE + RATTLE constraint algorithm.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use serde::Deserialize;

use crate::forces::{ConstraintList, GroupConstraint};
use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, constraint_virial_scatter, rattle_velocities,
    shake_positions, shake_positions_no_velocity, shake_snapshot,
};
use crate::kernels;
use crate::io::config::{ConfigError, NamedSlotConfig};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

use super::constraint::{Constraint, ConstraintBuilder, ConstraintError};
use crate::precision::Real;

pub const MAX_GROUP_ATOMS: u32 = 8;
pub const MAX_GROUP_CONSTRAINTS: u32 = 12;

// rq-f17b858f rq-811ba2a0
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShakeConstraintSpec {
    pub i: u32,
    pub j: u32,
    pub d: f64,
}

// rq-55f60603
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShakeParams {
    pub atoms: u32,
    pub constraints: Vec<ShakeConstraintSpec>,
}

fn deserialize_params(params: &toml::Value) -> Result<ShakeParams, ConfigError> {
    params
        .clone()
        .try_into::<ShakeParams>()
        .map_err(|e| crate::io::config::translate_params_error("constraint_types", e))
}

fn validate_shake_params(name: &str, p: &ShakeParams) -> Result<(), ConfigError> {
    if p.atoms == 0 {
        return Err(ConfigError::ShakeParamsMalformed {
            name: name.to_string(),
            reason: "atoms must be strictly positive".to_string(),
        });
    }
    if p.atoms > MAX_GROUP_ATOMS {
        return Err(ConfigError::ShakeParamsMalformed {
            name: name.to_string(),
            reason: format!(
                "atoms {} exceeds MAX_GROUP_ATOMS = {}",
                p.atoms, MAX_GROUP_ATOMS
            ),
        });
    }
    if p.constraints.is_empty() {
        return Err(ConfigError::ShakeParamsMalformed {
            name: name.to_string(),
            reason: "constraints list must not be empty".to_string(),
        });
    }
    if (p.constraints.len() as u32) > MAX_GROUP_CONSTRAINTS {
        return Err(ConfigError::ShakeParamsMalformed {
            name: name.to_string(),
            reason: format!(
                "constraints list length {} exceeds MAX_GROUP_CONSTRAINTS = {}",
                p.constraints.len(),
                MAX_GROUP_CONSTRAINTS
            ),
        });
    }
    let mut seen: Vec<(u32, u32)> = Vec::with_capacity(p.constraints.len());
    for c in &p.constraints {
        if c.i >= p.atoms || c.j >= p.atoms {
            return Err(ConfigError::ShakeParamsMalformed {
                name: name.to_string(),
                reason: format!(
                    "constraint (i={}, j={}) references out-of-range local atom (atoms = {})",
                    c.i, c.j, p.atoms
                ),
            });
        }
        if c.i == c.j {
            return Err(ConfigError::ShakeParamsMalformed {
                name: name.to_string(),
                reason: format!("constraint atoms must differ (i = j = {})", c.i),
            });
        }
        if !c.d.is_finite() || c.d <= 0.0 {
            return Err(ConfigError::ShakeParamsMalformed {
                name: name.to_string(),
                reason: format!(
                    "target distance must be strictly positive and finite, got {}",
                    c.d
                ),
            });
        }
        let key = if c.i < c.j { (c.i, c.j) } else { (c.j, c.i) };
        if seen.contains(&key) {
            return Err(ConfigError::ShakeParamsMalformed {
                name: name.to_string(),
                reason: format!("duplicate constraint pair ({}, {})", key.0, key.1),
            });
        }
        seen.push(key);
    }
    Ok(())
}

// rq-f17b858f rq-0b4600e2
#[derive(Debug, thiserror::Error)]
pub enum ShakeError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("shake constraint type `{name}` is malformed: {reason}")]
    MalformedShakeType { name: String, reason: String },
    #[error(
        "constraint group {group_index} shape ({atoms} atoms / {constraints} constraints) exceeds SHAKE per-group caps ({max_atoms} / {max_constraints}); use M-SHAKE for larger groups"
    )]
    UnsupportedGroupSize {
        group_index: usize,
        atoms: u32,
        constraints: u32,
        max_atoms: u32,
        max_constraints: u32,
    },
    #[error(
        "constraint group {group_index} has {actual_atoms} atoms but constraint type declares atoms = {expected_atoms}"
    )]
    GroupShapeMismatch {
        group_index: usize,
        expected_atoms: u32,
        actual_atoms: u32,
    },
}

impl From<ShakeError> for ConstraintError {
    fn from(e: ShakeError) -> Self {
        match e {
            ShakeError::Gpu(g) => ConstraintError::Gpu(g),
            ShakeError::Timings(t) => ConstraintError::Timings(t),
            ShakeError::MalformedShakeType { name, reason } => {
                ConstraintError::InvalidGroupShape {
                    group_index: 0,
                    kind: "shake".to_string(),
                    reason: format!("shake type {name}: {reason}"),
                }
            }
            ShakeError::UnsupportedGroupSize {
                group_index,
                atoms,
                constraints,
                max_atoms: _,
                max_constraints: _,
            } => ConstraintError::InvalidGroupShape {
                group_index,
                kind: "shake".to_string(),
                reason: format!(
                    "group shape ({atoms} atoms / {constraints} constraints) exceeds SHAKE per-group caps ({MAX_GROUP_ATOMS} / {MAX_GROUP_CONSTRAINTS})"
                ),
            },
            ShakeError::GroupShapeMismatch {
                group_index,
                expected_atoms,
                actual_atoms,
            } => ConstraintError::InvalidGroupShape {
                group_index,
                kind: "shake".to_string(),
                reason: format!(
                    "group has {actual_atoms} atoms but constraint type declares atoms = {expected_atoms}"
                ),
            },
        }
    }
}

// rq-d9a47c62
#[derive(Debug)]
pub struct ShakeConstraintsState {
    pub device: Arc<CudaDevice>,
    pub group_count: usize,
    pub particle_count: usize,
    pub atom_slot_count: usize,
    /// Largest atom count of any constraint group. Sizes the dynamic
    /// shared-memory staging buffer in `rattle_velocities`. rq-53800cef
    pub max_group_atoms: u32,
    pub group_atoms: CudaSlice<u32>,
    pub group_atom_offset: CudaSlice<u32>,
    pub group_atom_count: CudaSlice<u32>,
    pub group_constraint_offset: CudaSlice<u32>,
    pub group_constraint_count: CudaSlice<u32>,
    pub group_constraints_local_i: CudaSlice<u8>,
    pub group_constraints_local_j: CudaSlice<u8>,
    pub group_constraints_r2: CudaSlice<Real>,
    pub atom_mass: CudaSlice<Real>,
    pub snapshot_x: CudaSlice<Real>,
    pub snapshot_y: CudaSlice<Real>,
    pub snapshot_z: CudaSlice<Real>,
    pub constraint_virial: CudaSlice<Real>,
}

impl ShakeConstraintsState {
    pub fn new(
        device: Arc<CudaDevice>,
        list: &ConstraintList,
        masses: &[Real],
        constraint_types: &[NamedSlotConfig],
    ) -> Result<Self, ShakeError> {
        // Per-type validation: deserialise and bound-check every shake
        // type's params. Non-shake entries belong to other algorithms
        // and are skipped here; their builders validate them
        // separately.
        for ct in constraint_types {
            if ct.kind != "shake" {
                continue;
            }
            let p = ct
                .params
                .clone()
                .try_into::<ShakeParams>()
                .map_err(|e| ShakeError::MalformedShakeType {
                    name: ct.name.clone(),
                    reason: e.to_string(),
                })?;
            if let Err(e) = validate_shake_params(&ct.name, &p) {
                let reason = match e {
                    ConfigError::ShakeParamsMalformed { reason, .. } => reason,
                    other => other.to_string(),
                };
                return Err(ShakeError::MalformedShakeType {
                    name: ct.name.clone(),
                    reason,
                });
            }
        }

        // `ConstraintRegistry::build_optional` partitions the topology
        // by builder before calling here, so every group in `list`
        // belongs to a SHAKE-kind constraint type.
        let n_groups = list.groups.len();

        let mut group_atoms_host: Vec<u32> = Vec::new();
        let mut group_atom_offset_host: Vec<u32> = Vec::with_capacity(n_groups);
        let mut group_atom_count_host: Vec<u32> = Vec::with_capacity(n_groups);
        let mut group_constraint_offset_host: Vec<u32> = Vec::with_capacity(n_groups);
        let mut group_constraint_count_host: Vec<u32> = Vec::with_capacity(n_groups);
        let mut group_constraints_local_i_host: Vec<u8> = Vec::new();
        let mut group_constraints_local_j_host: Vec<u8> = Vec::new();
        let mut group_constraints_r2_host: Vec<Real> = Vec::new();

        for (gi, g) in list.groups.iter().enumerate() {
            let ct = &constraint_types[g.constraint_type_index as usize];
            let params: ShakeParams = ct
                .params
                .clone()
                .try_into()
                .map_err(|e| ShakeError::MalformedShakeType {
                    name: ct.name.clone(),
                    reason: e.to_string(),
                })?;
            // Cap check (defence-in-depth; validate_params bounds the
            // type-level params, but a group's actual shape comes from
            // the topology row and must match).
            if g.atom_count != params.atoms {
                return Err(ShakeError::GroupShapeMismatch {
                    group_index: gi,
                    expected_atoms: params.atoms,
                    actual_atoms: g.atom_count,
                });
            }
            if g.atom_count > MAX_GROUP_ATOMS || params.constraints.len() as u32 > MAX_GROUP_CONSTRAINTS {
                return Err(ShakeError::UnsupportedGroupSize {
                    group_index: gi,
                    atoms: g.atom_count,
                    constraints: params.constraints.len() as u32,
                    max_atoms: MAX_GROUP_ATOMS,
                    max_constraints: MAX_GROUP_CONSTRAINTS,
                });
            }

            let atom_offset = group_atoms_host.len() as u32;
            let constraint_offset = group_constraints_local_i_host.len() as u32;

            let atoms_slice = &list.group_atoms[g.atom_offset as usize
                ..(g.atom_offset + g.atom_count) as usize];
            group_atoms_host.extend_from_slice(atoms_slice);

            for c in &params.constraints {
                group_constraints_local_i_host.push(c.i as u8);
                group_constraints_local_j_host.push(c.j as u8);
                let d = c.d as Real;
                group_constraints_r2_host.push(d * d);
            }

            group_atom_offset_host.push(atom_offset);
            group_atom_count_host.push(g.atom_count);
            group_constraint_offset_host.push(constraint_offset);
            group_constraint_count_host.push(params.constraints.len() as u32);
        }

        let atom_slot_count = group_atoms_host.len();
        let max_group_atoms = group_atom_count_host.iter().copied().max().unwrap_or(0);

        // Per-atom mass array of length particle_count. Atoms not
        // referenced by any group are populated harmlessly with their
        // mass; the kernels only ever index into it via group_atoms.
        let atom_mass_host: Vec<Real> = masses.to_vec();

        let group_atoms = device
            .htod_sync_copy(&pad_min1(&group_atoms_host))
            .map_err(GpuError::from)?;
        let group_atom_offset = device
            .htod_sync_copy(&pad_min1(&group_atom_offset_host))
            .map_err(GpuError::from)?;
        let group_atom_count = device
            .htod_sync_copy(&pad_min1(&group_atom_count_host))
            .map_err(GpuError::from)?;
        let group_constraint_offset = device
            .htod_sync_copy(&pad_min1(&group_constraint_offset_host))
            .map_err(GpuError::from)?;
        let group_constraint_count = device
            .htod_sync_copy(&pad_min1(&group_constraint_count_host))
            .map_err(GpuError::from)?;
        let group_constraints_local_i = device
            .htod_sync_copy(&pad_min1_u8(&group_constraints_local_i_host))
            .map_err(GpuError::from)?;
        let group_constraints_local_j = device
            .htod_sync_copy(&pad_min1_u8(&group_constraints_local_j_host))
            .map_err(GpuError::from)?;
        let group_constraints_r2 = device
            .htod_sync_copy(&pad_min1_real(&group_constraints_r2_host))
            .map_err(GpuError::from)?;
        let atom_mass = if atom_mass_host.is_empty() {
            device.alloc_zeros::<Real>(1).map_err(GpuError::from)?
        } else {
            device.htod_sync_copy(&atom_mass_host).map_err(GpuError::from)?
        };

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

        Ok(ShakeConstraintsState {
            device,
            group_count: n_groups,
            particle_count: list.particle_count,
            atom_slot_count,
            max_group_atoms,
            group_atoms,
            group_atom_offset,
            group_atom_count,
            group_constraint_offset,
            group_constraint_count,
            group_constraints_local_i,
            group_constraints_local_j,
            group_constraints_r2,
            atom_mass,
            snapshot_x,
            snapshot_y,
            snapshot_z,
            constraint_virial,
        })
    }
}

fn pad_min1(v: &[u32]) -> Vec<u32> {
    if v.is_empty() { vec![0u32] } else { v.to_vec() }
}
fn pad_min1_u8(v: &[u8]) -> Vec<u8> {
    if v.is_empty() { vec![0u8] } else { v.to_vec() }
}
fn pad_min1_real(v: &[Real]) -> Vec<Real> {
    if v.is_empty() { vec![0.0] } else { v.to_vec() }
}

impl Constraint for ShakeConstraintsState {
    // rq-e538c545
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
        timings.kernel_start(KernelStage::SHAKE_SNAPSHOT)?;
        shake_snapshot(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &mut self.snapshot_x,
            &mut self.snapshot_y,
            &mut self.snapshot_z,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SHAKE_SNAPSHOT)?;
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
        timings.kernel_start(KernelStage::SHAKE_POSITIONS)?;
        shake_positions(
            buffers,
            &self.snapshot_x,
            &self.snapshot_y,
            &self.snapshot_z,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_constraint_offset,
            &self.group_constraint_count,
            &self.group_constraints_local_i,
            &self.group_constraints_local_j,
            &self.group_constraints_r2,
            &self.atom_mass,
            sim_box,
            dt,
            &mut self.constraint_virial,
            self.group_count,
            self.max_group_atoms,
        )?;
        timings.kernel_stop(KernelStage::SHAKE_POSITIONS)?;
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
        timings.kernel_start(KernelStage::RATTLE_VELOCITIES)?;
        rattle_velocities(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_constraint_offset,
            &self.group_constraint_count,
            &self.group_constraints_local_i,
            &self.group_constraints_local_j,
            &self.atom_mass,
            sim_box,
            dt,
            &mut self.constraint_virial,
            self.group_count,
            self.max_group_atoms,
        )?;
        timings.kernel_stop(KernelStage::RATTLE_VELOCITIES)?;
        timings.kernel_start(KernelStage::CONSTRAINT_VIRIAL_SCATTER)?;
        constraint_virial_scatter(
            buffers,
            &self.constraint_virial,
            &self.group_atoms,
            self.atom_slot_count,
        )?;
        timings.kernel_stop(KernelStage::CONSTRAINT_VIRIAL_SCATTER)?;
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
        timings.kernel_start(KernelStage::RATTLE_VELOCITIES)?;
        rattle_velocities(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_constraint_offset,
            &self.group_constraint_count,
            &self.group_constraints_local_i,
            &self.group_constraints_local_j,
            &self.atom_mass,
            sim_box,
            0.0,
            &mut self.constraint_virial,
            self.group_count,
            self.max_group_atoms,
        )?;
        timings.kernel_stop(KernelStage::RATTLE_VELOCITIES)?;
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
        timings.kernel_start(KernelStage::SHAKE_POSITIONS_NO_VELOCITY)?;
        shake_positions_no_velocity(
            buffers,
            &self.group_atoms,
            &self.group_atom_offset,
            &self.group_atom_count,
            &self.group_constraint_offset,
            &self.group_constraint_count,
            &self.group_constraints_local_i,
            &self.group_constraints_local_j,
            &self.group_constraints_r2,
            &self.atom_mass,
            sim_box,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SHAKE_POSITIONS_NO_VELOCITY)?;
        Ok(())
    }

    fn group_count(&self) -> usize {
        self.group_count
    }
}

// rq-c623013e
#[derive(Debug, Clone)]
pub struct ShakeBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for ShakeBuilder {
    fn kind_name(&self) -> &'static str {
        "shake"
    }}

impl ConstraintBuilder for ShakeBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        // Use a placeholder name; the registry-level error path
        // includes the real name when surfacing the error.
        validate_shake_params("<name placeholder>", &p)
    }

    fn expected_atom_count(&self, params: &toml::Value) -> usize {
        deserialize_params(params)
            .map(|p| p.atoms as usize)
            .unwrap_or(0)
    }

    fn expand_constraints(
        &self,
        params: &toml::Value,
    ) -> Result<Vec<GroupConstraint>, ConstraintError> {
        let p = deserialize_params(params).map_err(|e| ConstraintError::InvalidGroupShape {
            group_index: 0,
            kind: "shake".to_string(),
            reason: e.to_string(),
        })?;
        Ok(p.constraints
            .iter()
            .map(|c| GroupConstraint {
                local_i: c.i as u8,
                local_j: c.j as u8,
                r0: c.d as Real,
            })
            .collect())
    }

    fn validate_group_shape(
        &self,
        group_index: usize,
        atoms: &[u32],
        constraints: &[GroupConstraint],
        params: &toml::Value,
        _masses: &[Real],
    ) -> Result<(), ConstraintError> {
        let p = deserialize_params(params).map_err(|e| ConstraintError::InvalidGroupShape {
            group_index,
            kind: "shake".to_string(),
            reason: e.to_string(),
        })?;
        if atoms.len() as u32 != p.atoms {
            return Err(ConstraintError::InvalidGroupShape {
                group_index,
                kind: "shake".to_string(),
                reason: format!(
                    "{} atoms, expected {}",
                    atoms.len(),
                    p.atoms
                ),
            });
        }
        if constraints.len() != p.constraints.len() {
            return Err(ConstraintError::InvalidGroupShape {
                group_index,
                kind: "shake".to_string(),
                reason: format!(
                    "{} constraints, expected {}",
                    constraints.len(),
                    p.constraints.len()
                ),
            });
        }
        if atoms.len() as u32 > MAX_GROUP_ATOMS
            || (constraints.len() as u32) > MAX_GROUP_CONSTRAINTS
        {
            return Err(ConstraintError::InvalidGroupShape {
                group_index,
                kind: "shake".to_string(),
                reason: format!(
                    "shape ({} atoms / {} constraints) exceeds SHAKE per-group caps ({} / {})",
                    atoms.len(),
                    constraints.len(),
                    MAX_GROUP_ATOMS,
                    MAX_GROUP_CONSTRAINTS
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
        let state = ShakeConstraintsState::new(device, list, masses, constraint_types)?;
        Ok(Box::new(state))
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct ShakeKernels {
    pub shake_snapshot: CudaFunction,
    pub shake_positions: CudaFunction,
    pub rattle_velocities: CudaFunction,
    pub constraint_virial_scatter: CudaFunction,
    pub shake_positions_no_velocity: CudaFunction,
}

impl ShakeKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::SHAKE),
            "shake",
            &[
                "shake_snapshot",
                "shake_positions",
                "rattle_velocities",
                "constraint_virial_scatter",
                "shake_positions_no_velocity",
            ],
        )?;
        Ok(ShakeKernels {
            shake_snapshot: get_func(device, "shake", "shake_snapshot")?,
            shake_positions: get_func(device, "shake", "shake_positions")?,
            rattle_velocities: get_func(device, "shake", "rattle_velocities")?,
            constraint_virial_scatter: get_func(
                device,
                "shake",
                "constraint_virial_scatter",
            )?,
            shake_positions_no_velocity: get_func(
                device,
                "shake",
                "shake_positions_no_velocity",
            )?,
        })
    }
}
