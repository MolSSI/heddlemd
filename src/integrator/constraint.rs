// rq-3d5f2e98 — constraint slot framework.

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::forces::{ConstraintList, ConstraintTypeKind};
use crate::gpu::{GpuContext, GpuError, ParticleBuffers};
use crate::io::config::ConstraintTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{Timings, TimingsError};

// rq-7b1cdfb0
#[derive(Debug, thiserror::Error)]
pub enum ConstraintError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("constraint slot has no builder for algorithm kind {0:?}")]
    UnsupportedKind(ConstraintTypeKind),
    #[error("[constraints] row references unknown constraint type `{0}`")]
    UnknownConstraintType(String),
    #[error("atom {atom} appears in more than one constraint group")]
    DuplicateConstraintAtom { atom: u32 },
    #[error(
        "constraint group {group_index} ({kind:?}) has invalid shape: {reason}"
    )]
    InvalidGroupShape {
        group_index: usize,
        kind: ConstraintTypeKind,
        reason: String,
    },
    #[error(
        "pair (atoms {atom_i}, {atom_j}) appears in both a bond and a constraint"
    )]
    ConstraintBondPairOverlap { atom_i: u32, atom_j: u32 },
}

// rq-f08d7a33 — Constraint trait. Hooks fire at sub-step boundaries
// inside the integrator (see `constraint-framework.md`).
pub trait Constraint: std::fmt::Debug + Send {
    /// Snapshot pre-drift positions for the slot's owned atoms.
    fn apply_before_drift(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError>;

    /// Project unconstrained post-drift positions onto the constraint
    /// manifold, updating half-step velocities for consistency.
    fn apply_after_drift(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError>;

    /// Project the integrator's final velocities onto the constraint
    /// manifold (the time-derivative of every constraint is zero at the
    /// new positions).
    fn apply_after_kick(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ConstraintError>;

    /// Number of constraint groups the slot owns. Tests use this to
    /// assert empty-state behaviour.
    fn group_count(&self) -> usize {
        0
    }
}

// rq-3d5f2e98 — builder trait. Concrete slots register a builder in
// `ConstraintRegistry::with_builtins()`.
pub trait ConstraintBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        device: Arc<CudaDevice>,
        gpu: &GpuContext,
        particle_count: usize,
        list: &ConstraintList,
        masses: &[f32],
        constraint_types: &[ConstraintTypeConfig],
    ) -> Result<Box<dyn Constraint>, ConstraintError>;
}

// rq-3d5f2e98
#[derive(Debug)]
pub struct ConstraintRegistry {
    pub builders: Vec<Box<dyn ConstraintBuilder>>,
}

impl ConstraintRegistry {
    pub fn new() -> Self {
        ConstraintRegistry {
            builders: Vec::new(),
        }
    }

    pub fn with_builtins() -> Self {
        ConstraintRegistry {
            builders: vec![Box::new(crate::integrator::settle::SettleBuilder)],
        }
    }

    pub fn register(&mut self, builder: Box<dyn ConstraintBuilder>) {
        self.builders.push(builder);
    }

    /// Construct the constraint slot, if any, that handles every group in
    /// `list`. v1 produces a single slot implementation covering every
    /// supported algorithm kind referenced by `list`. Returns `Ok(None)`
    /// when `list.is_empty()`.
    pub fn build_optional(
        &self,
        list: &ConstraintList,
        gpu: &GpuContext,
        particle_count: usize,
        masses: &[f32],
        constraint_types: &[ConstraintTypeConfig],
    ) -> Result<Option<Box<dyn Constraint>>, ConstraintError> {
        if list.is_empty() {
            return Ok(None);
        }
        // v1: every group's kind must be SettleWater (the only registered
        // algorithm). Future M-SHAKE work introduces a second kind and a
        // dispatch wrapper that combines the per-algorithm slots.
        for g in &list.groups {
            let kind = list.constraint_type_kind[g.constraint_type_index as usize];
            let builder_name = match kind {
                ConstraintTypeKind::SettleWater => "settle",
            };
            if !self.builders.iter().any(|b| b.kind_name() == builder_name) {
                return Err(ConstraintError::UnsupportedKind(kind));
            }
        }
        // In v1 there is exactly one builder ("settle") and it consumes
        // every group. When M-SHAKE arrives, this dispatch will fan out
        // to per-algorithm builders and combine their results.
        let settle = self
            .builders
            .iter()
            .find(|b| b.kind_name() == "settle")
            .ok_or(ConstraintError::UnsupportedKind(ConstraintTypeKind::SettleWater))?;
        let slot = settle.build(
            gpu.device.clone(),
            gpu,
            particle_count,
            list,
            masses,
            constraint_types,
        )?;
        Ok(Some(slot))
    }
}

impl Default for ConstraintRegistry {
    fn default() -> Self {
        ConstraintRegistry::with_builtins()
    }
}
