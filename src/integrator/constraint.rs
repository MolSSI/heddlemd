// rq-3d5f2e98 — constraint slot framework.

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::forces::{ConstraintList, GroupConstraint};
use crate::gpu::{GpuContext, GpuError, ParticleBuffers};
use crate::io::config::{ConfigError, NamedSlotConfig};
use crate::pbc::SimulationBox;
use crate::timings::{Timings, TimingsError};

// rq-7b1cdfb0
#[derive(Debug, thiserror::Error)]
pub enum ConstraintError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("constraint slot has no builder for algorithm kind `{0}`")]
    UnsupportedKind(String),
    #[error("[constraints] row references unknown constraint type `{0}`")]
    UnknownConstraintType(String),
    #[error("atom {atom} appears in more than one constraint group")]
    DuplicateConstraintAtom { atom: u32 },
    #[error("constraint group {group_index} (kind `{kind}`) has invalid shape: {reason}")]
    InvalidGroupShape {
        group_index: usize,
        kind: String,
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

    /// Project the runner's freshly-sampled initial velocities onto
    /// the constraint velocity manifold. Called once by the runner
    /// after the initial Maxwell-Boltzmann sample and before the
    /// first integrator step, so the system starts the run already
    /// on the manifold (rather than relying on the first step's
    /// `apply_after_kick` to do it, which would drop ~`n_constraints /
    /// 3N` of the sampled kinetic energy and leave the integrator
    /// starting at the wrong displayed temperature).
    ///
    /// The default does nothing (no projection needed for an empty
    /// constraint list); algorithms that own a velocity projection
    /// override this. Implementations must not touch
    /// `buffers.forces_*`, `buffers.potential_energies`, or
    /// `buffers.virials`.
    fn apply_initial_velocity_projection(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _sim_box: &SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        Ok(())
    }

    /// Project per-particle positions back onto the constraint
    /// manifold without modifying velocities, virials, or any other
    /// buffer. Driven by the minimization runner (see
    /// `rqm/minimization/steepest-descent.md`); never called from the
    /// integration plan walk.
    ///
    /// Implementations must mutate only `buffers.positions_*`; they
    /// must not consume `dt` (minimization has no time scale). The
    /// default returns `Ok(())` for slots that own no groups.
    fn apply_position_projection_only(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _sim_box: &SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        Ok(())
    }

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

    /// Validate the kind-specific parameters of a
    /// `[[constraint_types]]` entry at config-load time. Called by
    /// `Config::validate_against(&registries)` before any GPU work.
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>;

    /// `true` iff the algorithm implements
    /// `Constraint::apply_position_projection_only` non-trivially
    /// (i.e., can participate in minimization phases). The default
    /// returns `true`. Algorithms that cannot project positions
    /// without a paired velocity / virial update override this to
    /// return `false`; configs that pair such an algorithm with a
    /// `[[minimization]]` phase are rejected at config load via
    /// `Config::validate_constraint_compatibility`.
    fn supports_position_projection_only(&self, _params: &toml::Value) -> bool {
        true
    }

    /// Number of atoms a single `[constraints]` topology row of this
    /// kind must declare. The topology parser uses this value to
    /// validate row column counts. Pure function of the parameters.
    fn expected_atom_count(&self, params: &toml::Value) -> usize;

    /// Validate the cluster shape of a single constraint group against
    /// this algorithm's requirements (atom count, constraint-pair
    /// pattern, mass consistency, etc.). Called by
    /// `ConstraintRegistry::build_optional` for every group whose
    /// algorithm matches this builder.
    fn validate_group_shape(
        &self,
        group_index: usize,
        atoms: &[u32],
        constraints: &[GroupConstraint],
        params: &toml::Value,
        masses: &[f32],
    ) -> Result<(), ConstraintError>;

    fn build(
        &self,
        device: Arc<CudaDevice>,
        gpu: &GpuContext,
        particle_count: usize,
        list: &ConstraintList,
        masses: &[f32],
        constraint_types: &[NamedSlotConfig],
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

    pub fn lookup(&self, kind: &str) -> Option<&dyn ConstraintBuilder> {
        for b in &self.builders {
            if b.kind_name() == kind {
                return Some(b.as_ref());
            }
        }
        None
    }

    /// Construct the constraint slot, if any, that handles every group
    /// in `list`. v1 produces a single slot implementation covering
    /// every supported algorithm kind referenced by `list`. Returns
    /// `Ok(None)` when `list.is_empty()`.
    ///
    /// For every group, looks up the algorithm via
    /// `constraint_types[group.constraint_type_index].kind`, finds the
    /// matching builder, calls `validate_group_shape(...)`, and
    /// finally delegates to the slot constructor.
    pub fn build_optional(
        &self,
        list: &ConstraintList,
        gpu: &GpuContext,
        particle_count: usize,
        masses: &[f32],
        constraint_types: &[NamedSlotConfig],
    ) -> Result<Option<Box<dyn Constraint>>, ConstraintError> {
        if list.is_empty() {
            return Ok(None);
        }
        // First, verify every group's algorithm is registered.
        for g in &list.groups {
            let kind = &constraint_types[g.constraint_type_index as usize].kind;
            if self.lookup(kind).is_none() {
                return Err(ConstraintError::UnsupportedKind(kind.clone()));
            }
        }
        // Run per-builder validate_group_shape on every group.
        for (gi, g) in list.groups.iter().enumerate() {
            let cfg = &constraint_types[g.constraint_type_index as usize];
            let builder = self
                .lookup(&cfg.kind)
                .ok_or_else(|| ConstraintError::UnsupportedKind(cfg.kind.clone()))?;
            let atoms = &list.group_atoms[g.atom_offset as usize
                ..(g.atom_offset + g.atom_count) as usize];
            let cstrs = &list.group_constraints[g.constraint_offset as usize
                ..(g.constraint_offset + g.constraint_count) as usize];
            builder.validate_group_shape(gi, atoms, cstrs, &cfg.params, masses)?;
        }
        // v1: every group is "settle-water"; one builder consumes all
        // groups. When M-SHAKE arrives, this dispatch fans out per
        // algorithm and the per-algorithm slots are combined.
        let settle = self
            .lookup("settle-water")
            .ok_or_else(|| ConstraintError::UnsupportedKind("settle-water".to_string()))?;
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
