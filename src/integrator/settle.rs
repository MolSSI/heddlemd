// rq-67e62f4b — SETTLE analytic three-atom rigid-water constraint slot.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::forces::{ConstraintList, ConstraintTypeKind};
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, settle_positions, settle_snapshot, settle_velocities,
};
use crate::io::config::ConstraintTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

use super::constraint::{Constraint, ConstraintBuilder, ConstraintError};

// rq-67e62f4b
#[derive(Debug, thiserror::Error)]
pub enum SettleError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error(
        "constraint group {group_index} has inconsistent SETTLE masses: expected (O={expected_o}, H={expected_h}); got ({actual_o}, {actual_h1}, {actual_h2})"
    )]
    InconsistentMasses {
        group_index: usize,
        expected_o: f32,
        expected_h: f32,
        actual_o: f32,
        actual_h1: f32,
        actual_h2: f32,
    },
    #[error("settle-water constraint type `{name}` is malformed: {reason}")]
    MalformedSettleType { name: String, reason: String },
    #[error(
        "settle group at index {group_index} has shape {actual_atoms} atoms / {actual_constraints} constraints; expected 3 / 3"
    )]
    InvalidGroupShape {
        group_index: usize,
        actual_atoms: u32,
        actual_constraints: u32,
    },
}

impl From<SettleError> for ConstraintError {
    fn from(e: SettleError) -> Self {
        match e {
            SettleError::Gpu(g) => ConstraintError::Gpu(g),
            SettleError::Timings(t) => ConstraintError::Timings(t),
            SettleError::InconsistentMasses { group_index, .. } => {
                ConstraintError::InvalidGroupShape {
                    group_index,
                    kind: ConstraintTypeKind::SettleWater,
                    reason: format!("{e}"),
                }
            }
            SettleError::MalformedSettleType { name, reason } => {
                ConstraintError::InvalidGroupShape {
                    group_index: 0,
                    kind: ConstraintTypeKind::SettleWater,
                    reason: format!("settle type {name}: {reason}"),
                }
            }
            SettleError::InvalidGroupShape {
                group_index,
                actual_atoms,
                actual_constraints,
            } => ConstraintError::InvalidGroupShape {
                group_index,
                kind: ConstraintTypeKind::SettleWater,
                reason: format!(
                    "{actual_atoms} atoms / {actual_constraints} constraints, expected 3 / 3"
                ),
            },
        }
    }
}

// rq-67e62f4b
#[derive(Debug)]
pub struct SettleConstraintsState {
    pub device: Arc<CudaDevice>,
    pub group_count: usize,
    pub particle_count: usize,
    pub group_atoms: CudaSlice<u32>,
    pub group_type_index: CudaSlice<u32>,
    pub type_canonical_x: CudaSlice<f32>,
    pub type_canonical_y: CudaSlice<f32>,
    pub type_canonical_z: CudaSlice<f32>,
    pub type_mass_o: CudaSlice<f32>,
    pub type_mass_h: CudaSlice<f32>,
    pub snapshot_x: CudaSlice<f32>,
    pub snapshot_y: CudaSlice<f32>,
    pub snapshot_z: CudaSlice<f32>,
}

impl SettleConstraintsState {
    /// Build the slot from the topology's `ConstraintList`, the per-atom
    /// masses pulled from the `ParticleState`, and the full
    /// `[[constraint_types]]` config table. v1 expects every group's
    /// `constraint_type_kind` to be `SettleWater`; the framework's
    /// `build_optional` enforces this before calling here.
    pub fn new(
        device: Arc<CudaDevice>,
        list: &ConstraintList,
        masses: &[f32],
        constraint_types: &[ConstraintTypeConfig],
    ) -> Result<Self, SettleError> {
        // Validate group shapes and pack the group-atoms flat array in
        // group order. SETTLE requires exactly 3 atoms and 3
        // constraints per group, in the (0,1)/(0,2)/(1,2) pattern.
        let n_groups = list.groups.len();
        let mut group_atoms_host: Vec<u32> = Vec::with_capacity(3 * n_groups);
        let mut group_type_index_host: Vec<u32> = Vec::with_capacity(n_groups);
        for (gi, g) in list.groups.iter().enumerate() {
            if g.atom_count != 3 || g.constraint_count != 3 {
                return Err(SettleError::InvalidGroupShape {
                    group_index: gi,
                    actual_atoms: g.atom_count,
                    actual_constraints: g.constraint_count,
                });
            }
            let atoms = &list.group_atoms[g.atom_offset as usize
                ..(g.atom_offset + g.atom_count) as usize];
            // Verify the (0,1), (0,2), (1,2) pattern; the parser
            // currently emits constraints in this order but we re-check
            // explicitly so the contract stays decoupled from the
            // parser's internal ordering.
            let cstrs = &list.group_constraints[g.constraint_offset as usize
                ..(g.constraint_offset + g.constraint_count) as usize];
            let mut seen: [bool; 3] = [false; 3];
            for c in cstrs {
                let pair = (c.local_i.min(c.local_j), c.local_i.max(c.local_j));
                let idx = match pair {
                    (0, 1) => 0,
                    (0, 2) => 1,
                    (1, 2) => 2,
                    _ => {
                        return Err(SettleError::InvalidGroupShape {
                            group_index: gi,
                            actual_atoms: g.atom_count,
                            actual_constraints: g.constraint_count,
                        });
                    }
                };
                seen[idx] = true;
            }
            if !(seen[0] && seen[1] && seen[2]) {
                return Err(SettleError::InvalidGroupShape {
                    group_index: gi,
                    actual_atoms: g.atom_count,
                    actual_constraints: g.constraint_count,
                });
            }
            group_atoms_host.extend_from_slice(atoms);
            group_type_index_host.push(g.constraint_type_index);
        }

        // Per-type canonical body-frame positions and (m_O, m_H)
        // tables, indexed by the order of `constraint_types`. Each
        // entry's `kind` must be SettleWater (the framework's
        // `build_optional` guarantees this for v1).
        let n_types = constraint_types.len();
        let mut type_canonical_x_host: Vec<f32> = Vec::with_capacity(3 * n_types);
        let mut type_canonical_y_host: Vec<f32> = Vec::with_capacity(3 * n_types);
        let mut type_canonical_z_host: Vec<f32> = Vec::with_capacity(3 * n_types);
        let mut type_mass_o_host: Vec<f32> = vec![0.0; n_types];
        let mut type_mass_h_host: Vec<f32> = vec![0.0; n_types];
        for (_ti, ct) in constraint_types.iter().enumerate() {
            match ct {
                ConstraintTypeConfig::SettleWater { name, r_oh, r_hh } => {
                    let r_oh = *r_oh as f32;
                    let r_hh = *r_hh as f32;
                    if r_hh >= 2.0 * r_oh {
                        return Err(SettleError::MalformedSettleType {
                            name: name.clone(),
                            reason: format!("r_hh {r_hh} >= 2 r_oh {}", 2.0 * r_oh),
                        });
                    }
                    // Canonical geometry with mass-weighted COM at the
                    // origin. We place the molecule in the xy-plane
                    // with H-H along y and O on the +x axis. The
                    // distance from O to the H-H midpoint is
                    //   d_oh = sqrt(r_oh² - (r_hh/2)²)
                    // and the (mass-weighted) COM displacement from
                    // the midpoint along x is m_o · d_oh / total_mass
                    // (taken from the placeholder masses until the
                    // first group that references this type provides
                    // the actual m_O, m_H — we revisit below).
                    let d_oh = (r_oh * r_oh - (r_hh * 0.5) * (r_hh * 0.5)).sqrt();
                    // Defer COM-anchoring until per-group masses are
                    // applied below; store the geometry in the H-H
                    // midpoint frame for now.
                    type_canonical_x_host.push(d_oh); // O on +x
                    type_canonical_y_host.push(0.0);
                    type_canonical_z_host.push(0.0);
                    type_canonical_x_host.push(0.0); // H1 at (0, -r_hh/2, 0)
                    type_canonical_y_host.push(-r_hh * 0.5);
                    type_canonical_z_host.push(0.0);
                    type_canonical_x_host.push(0.0); // H2 at (0, +r_hh/2, 0)
                    type_canonical_y_host.push(r_hh * 0.5);
                    type_canonical_z_host.push(0.0);
                }
            }
        }

        // Validate masses for every group and derive the per-type
        // (m_O, m_H) table. Two groups of the same constraint type
        // must agree on (m_O, m_H, m_H); otherwise SETTLE's analytic
        // closed form does not apply.
        let mut type_mass_o_set: Vec<Option<f32>> = vec![None; n_types];
        let mut type_mass_h_set: Vec<Option<f32>> = vec![None; n_types];
        for (gi, g) in list.groups.iter().enumerate() {
            let ti = g.constraint_type_index as usize;
            let atoms = &list.group_atoms[g.atom_offset as usize
                ..(g.atom_offset + g.atom_count) as usize];
            let m_o = masses[atoms[0] as usize];
            let m_h1 = masses[atoms[1] as usize];
            let m_h2 = masses[atoms[2] as usize];
            // Pure relative tolerance: m_h1 and m_h2 are physical atomic
            // masses and may be tiny (~1e-27 kg). Adding `.max(1.0)`
            // would turn the comparison into a near-absolute one that
            // never fires for realistic inputs.
            let mh_scale = m_h1.abs().max(m_h2.abs());
            if mh_scale > 0.0 && (m_h1 - m_h2).abs() > 1.0e-6 * mh_scale {
                return Err(SettleError::InconsistentMasses {
                    group_index: gi,
                    expected_o: type_mass_o_set[ti].unwrap_or(m_o),
                    expected_h: type_mass_h_set[ti].unwrap_or(m_h1),
                    actual_o: m_o,
                    actual_h1: m_h1,
                    actual_h2: m_h2,
                });
            }
            match (type_mass_o_set[ti], type_mass_h_set[ti]) {
                (None, None) => {
                    type_mass_o_set[ti] = Some(m_o);
                    type_mass_h_set[ti] = Some(m_h1);
                }
                (Some(exp_o), Some(exp_h)) => {
                    let tol_o = 1.0e-6 * exp_o.abs().max(m_o.abs());
                    let tol_h = 1.0e-6 * exp_h.abs().max(m_h1.abs());
                    if (exp_o - m_o).abs() > tol_o || (exp_h - m_h1).abs() > tol_h {
                        return Err(SettleError::InconsistentMasses {
                            group_index: gi,
                            expected_o: exp_o,
                            expected_h: exp_h,
                            actual_o: m_o,
                            actual_h1: m_h1,
                            actual_h2: m_h2,
                        });
                    }
                }
                _ => unreachable!(),
            }
        }
        for (ti, (mo, mh)) in type_mass_o_set.iter().zip(type_mass_h_set.iter()).enumerate() {
            type_mass_o_host[ti] = mo.unwrap_or(0.0);
            type_mass_h_host[ti] = mh.unwrap_or(0.0);
        }

        // Now anchor the canonical body-frame positions so the
        // mass-weighted centroid is at the origin: shift x by
        // -m_O · d_oh / M for every type. (Skip types unused by any
        // group; their canonical values are irrelevant.)
        for ti in 0..n_types {
            let m_o = type_mass_o_host[ti];
            let m_h = type_mass_h_host[ti];
            let total = m_o + 2.0 * m_h;
            if total <= 0.0 {
                continue;
            }
            // d_oh is currently in type_canonical_x[3*ti+0]; the H
            // entries have x=0, so shifting all three by -m_O·d_oh/M
            // re-anchors the COM at the origin.
            let d_oh = type_canonical_x_host[3 * ti];
            let shift = -m_o * d_oh / total;
            // Equivalent to: O_x -= m_o · d_oh / M
            //                 H1_x, H2_x: also shifted (they were 0,
            //                 now become +shift_h = m_h · d_oh / M
            //                 ... but only when expressed COM-centered)
            // Wait — after shifting all atoms by `shift`, the COM is
            //   m_o (d_oh + shift) + 2 m_h (0 + shift)
            // = m_o d_oh + (m_o + 2 m_h) shift
            // = m_o d_oh - m_o d_oh = 0. Correct.
            type_canonical_x_host[3 * ti] += shift;
            type_canonical_x_host[3 * ti + 1] += shift;
            type_canonical_x_host[3 * ti + 2] += shift;
        }

        // Upload buffers.
        let group_atoms = device
            .htod_sync_copy(&group_atoms_host)
            .map_err(GpuError::from)?;
        let group_type_index = device
            .htod_sync_copy(&group_type_index_host)
            .map_err(GpuError::from)?;
        let type_canonical_x = device
            .htod_sync_copy(&type_canonical_x_host)
            .map_err(GpuError::from)?;
        let type_canonical_y = device
            .htod_sync_copy(&type_canonical_y_host)
            .map_err(GpuError::from)?;
        let type_canonical_z = device
            .htod_sync_copy(&type_canonical_z_host)
            .map_err(GpuError::from)?;
        let type_mass_o = device
            .htod_sync_copy(&type_mass_o_host)
            .map_err(GpuError::from)?;
        let type_mass_h = device
            .htod_sync_copy(&type_mass_h_host)
            .map_err(GpuError::from)?;
        let snapshot_x = device
            .alloc_zeros::<f32>(3 * n_groups)
            .map_err(GpuError::from)?;
        let snapshot_y = device
            .alloc_zeros::<f32>(3 * n_groups)
            .map_err(GpuError::from)?;
        let snapshot_z = device
            .alloc_zeros::<f32>(3 * n_groups)
            .map_err(GpuError::from)?;

        Ok(SettleConstraintsState {
            device,
            group_count: n_groups,
            particle_count: list.particle_count,
            group_atoms,
            group_type_index,
            type_canonical_x,
            type_canonical_y,
            type_canonical_z,
            type_mass_o,
            type_mass_h,
            snapshot_x,
            snapshot_y,
            snapshot_z,
        })
    }
}

impl Constraint for SettleConstraintsState {
    fn apply_before_drift(
        &mut self,
        buffers: &mut ParticleBuffers,
        _sim_box: &SimulationBox,
        _dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        if self.group_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SETTLE_SNAPSHOT)?;
        settle_snapshot(
            buffers,
            &self.group_atoms,
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
        dt: f32,
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
            &self.group_type_index,
            &self.type_canonical_x,
            &self.type_canonical_y,
            &self.type_canonical_z,
            &self.type_mass_o,
            &self.type_mass_h,
            sim_box,
            dt,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_POSITIONS)?;
        Ok(())
    }

    fn apply_after_kick(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        _dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        if self.group_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SETTLE_VELOCITIES)?;
        settle_velocities(
            buffers,
            &self.group_atoms,
            &self.group_type_index,
            &self.type_mass_o,
            &self.type_mass_h,
            sim_box,
            self.group_count,
        )?;
        timings.kernel_stop(KernelStage::SETTLE_VELOCITIES)?;
        Ok(())
    }

    fn group_count(&self) -> usize {
        self.group_count
    }
}

#[derive(Debug)]
pub struct SettleBuilder;

impl ConstraintBuilder for SettleBuilder {
    fn kind_name(&self) -> &'static str {
        "settle"
    }

    fn build(
        &self,
        device: Arc<CudaDevice>,
        _gpu: &GpuContext,
        _particle_count: usize,
        list: &ConstraintList,
        masses: &[f32],
        constraint_types: &[ConstraintTypeConfig],
    ) -> Result<Box<dyn Constraint>, ConstraintError> {
        let state = SettleConstraintsState::new(device, list, masses, constraint_types)?;
        Ok(Box::new(state))
    }
}
