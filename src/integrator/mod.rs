// rq-e0a0553d rq-6cd635cd rq-6c5b4246
//
// Three orthogonal slot frameworks: integrator, thermostat, barostat.
// The runner chains the slots `apply_pre → step → apply_post → apply`
// per timestep (see `simulation-runner.md` and `framework.md`).

use crate::forces::{ForceField, ForceFieldError};
use crate::gpu::{GpuContext, GpuError, ParticleBuffers};
use crate::io::config::{ConfigError, SlotConfig};
use crate::registry::{Builtins, KindedBuilder, Registry};
use crate::pbc::SimulationBox;
use crate::timings::{Timings, TimingsError};
use crate::precision::Real;

pub mod andersen;
pub mod berendsen;
pub mod berendsen_barostat;
pub mod c_rescale_barostat;
pub mod constraint;
pub mod csvr;
pub mod langevin_baoab;
pub mod mtk_npt;
pub mod nose_hoover_chain;
pub mod philox;
pub mod settle;
pub mod shake;
pub mod velocity_verlet;

pub use andersen::{AndersenBuilder, AndersenThermostat};
pub use berendsen::{BerendsenBuilder, BerendsenThermostat};
pub use berendsen_barostat::{BerendsenBarostat, BerendsenBarostatBuilder};
pub use c_rescale_barostat::{CRescaleBarostat, CRescaleBarostatBuilder};
pub use constraint::{Constraint, ConstraintBuilder, ConstraintError, ConstraintRegistry};
pub use csvr::{CsvrBuilder, CsvrThermostat};
pub use langevin_baoab::{LangevinBaoabBuilder, LangevinBaoabState};
pub use mtk_npt::{MtkNptBuilder, MtkNptIntegrator};
pub use nose_hoover_chain::{
    NoseHooverChainBuilder, NoseHooverChainThermostat, nhc_chain_sub_step,
};
pub use philox::{philox_4x32_10, philox_normal};
pub use settle::{SettleBuilder, SettleConstraintsState, SettleError};
pub use shake::{ShakeBuilder, ShakeConstraintsState, ShakeError};
pub use velocity_verlet::{VelocityVerletBuilder, VelocityVerletState};

// rq-df6d79a1
pub use crate::forces::ForceClass;

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum IntegratorError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("unknown integrator kind `{0}`")]
    UnknownKind(String),
    #[error("integrator's execute() received unsupported sub-step variant {variant}")]
    UnexpectedSubStep { variant: &'static str },
}

// rq-52e52d7b
/// Unified error returned by [`run_step`]: the plan walker can surface
/// failures from the integrator's `execute()`, from the runner-dispatched
/// `force_field.step(...)`, or from any constraint hook.
#[derive(Debug, thiserror::Error)]
pub enum StepError {
    #[error("{0}")]
    Integrator(#[from] IntegratorError),
    #[error("{0}")]
    ForceField(#[from] ForceFieldError),
    #[error("{0}")]
    Constraint(#[from] ConstraintError),
    // rq-0e26dde0
    /// Returned by `IntegratorStepWithConstraintExt::step_with_constraint`
    /// when the integrator's `check_accepts_constraints_now()`
    /// rejected hook installation for the instance's current runtime
    /// state (e.g., velocity-Verlet with `lossless = true`). The
    /// `reason` is the integrator's verbatim message.
    #[error("integrator rejected constraint hook installation: {reason}")]
    IntegratorRejectsConstraint { reason: &'static str },
    #[error(
        "built-in {kind} slot `{label}` did not expose a post-force per-particle \
         source fragment via post_force_per_particle_fragment"
    )]
    MissingPostForcePerParticleFragment {
        kind: &'static str,
        label: &'static str,
    },
    #[error("JIT-composed post-force per-particle kernel failed to compile: {log}")]
    PostForceFragmentCompileFailed { log: String },
    #[error("JIT-composed post-force per-particle kernel failed to load: {0}")]
    PostForceFragmentLoadFailed(GpuError),
}

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum ThermostatError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("unknown thermostat kind `{0}`")]
    UnknownKind(String),
}

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum BarostatError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("unknown barostat kind `{0}`")]
    UnknownKind(String),
}

// --- Integrator trait, builder, registry ------------------------------

// rq-dbbffa7d
/// One piece of an integrator's per-timestep work, described in the
/// `StepPlan` returned by [`Integrator::plan`].
#[derive(Debug, Clone, Copy)]
pub enum SubStep {
    /// Velocity half-kick: `v ← v + (F/m) · dt/2` (or the integrator's
    /// equivalent). No position update.
    KickHalf { dt: Real, label: &'static str },
    /// Position drift: `x ← x + v · dt` (or the integrator's
    /// equivalent). No velocity update.
    Drift { dt: Real, label: &'static str },
    /// Fused KickHalf + Drift in a single kernel launch
    /// (e.g. `vv_kick_drift`).
    KickDrift { dt: Real, label: &'static str },
    /// Force-pipeline evaluation. Dispatched by the runner, not by
    /// the integrator's `execute()`. `class` selects which force
    /// class(es) to re-evaluate:
    /// - `None` → runner calls `force_field.step(...)` (every slot).
    /// - `Some(class)` → runner calls
    ///   `force_field.step_class(class, ...)` (only matching slots).
    /// In both cases the combiner refreshes
    /// `ParticleBuffers.forces_*` from every class's slot-output
    /// buffers.
    ForceEval {
        class: Option<crate::forces::ForceClass>,
        /// Aggregation level the integrator needs at this sub-step.
        ///
        /// - `Some(ForcesAndScalars)` → integrator needs fresh energy
        ///   and virial (e.g. NPT barostats reading virial every step).
        /// - `Some(ForcesOnly)` → integrator only needs forces.
        /// - `None` → no preference; the runner picks based on its own
        ///   needs (logging cadence, trajectory cadence, minimization).
        ///
        /// The runner's `resolve_level` upgrades any request to
        /// `ForcesAndScalars` whenever it independently needs scalars
        /// at this step.
        level: Option<crate::forces::AggregateLevel>,
    },
    /// Integrator-private sub-step (e.g. Langevin's OU step, MTK's
    /// chain or barostat sub-steps). `dt` carries the outer plan
    /// timestep so the integrator's `execute()` can compute its
    /// substep-specific factors without needing to cache `dt` in
    /// `&mut self`; the `label` lets `execute()` dispatch to the right
    /// kernel.
    Custom { dt: Real, label: &'static str },
}

impl SubStep {
    /// Returns the variant name (without the payload) as a static
    /// string. Useful for error reporting and the runner's hook-position
    /// inference.
    pub fn variant_name(&self) -> &'static str {
        match self {
            SubStep::KickHalf { .. } => "KickHalf",
            SubStep::Drift { .. } => "Drift",
            SubStep::KickDrift { .. } => "KickDrift",
            SubStep::ForceEval { .. } => "ForceEval",
            SubStep::Custom { .. } => "Custom",
        }
    }

    /// True iff a constraint slot's `apply_before_drift` /
    /// `apply_after_drift` hooks should fire around this sub-step.
    pub fn is_drift(&self) -> bool {
        matches!(self, SubStep::Drift { .. } | SubStep::KickDrift { .. })
    }

    /// True iff a constraint slot's `apply_after_kick` hook should fire
    /// after this sub-step when it is the final sub-step of the plan.
    pub fn is_velocity_update(&self) -> bool {
        matches!(self, SubStep::KickHalf { .. } | SubStep::KickDrift { .. })
    }
}

// rq-9fbba3be
/// Ordered list of sub-steps that constitute one full timestep.
#[derive(Debug, Clone)]
pub struct StepPlan {
    pub steps: Vec<SubStep>,
}

impl StepPlan {
    pub fn empty() -> Self {
        StepPlan { steps: Vec::new() }
    }
}

// rq-4187d20f
/// Capability trait carrying both an integrator / thermostat / barostat
/// slot's post-force per-particle fragment and its launch-time argument
/// binding, so a slot cannot provide one without the other. A slot that
/// participates returns `Some(self)` from its trait's
/// `post_force_per_particle` accessor. See
/// `rqm/integration/jit-composed-post-force.md`.
pub trait PostForcePerParticle {
    fn post_force_per_particle_fragment(&self) -> crate::forces::PerParticleFragment;

    fn bind_post_force_per_particle_args(
        &self,
        ctx: &crate::forces::PostForceBindContext<'_>,
        builder: &mut crate::forces::ForceLaunchBuilder,
    );
}

// rq-78f484d9
pub trait Integrator: std::fmt::Debug + Send {
    /// Return the ordered sequence of sub-steps that constitute one
    /// timestep of size `dt`. Pure: must return the same shape for the
    /// same `dt` and the same integrator state across calls.
    fn plan(&self, dt: Real) -> StepPlan;

    /// Execute one sub-step from this integrator's plan. The runner
    /// calls this for every sub-step EXCEPT `SubStep::ForceEval`, which
    /// the runner dispatches directly via `force_field.step(...)`. An
    /// integrator that receives `SubStep::ForceEval` here returns
    /// `IntegratorError::UnexpectedSubStep`.
    fn execute(
        &mut self,
        substep: &SubStep,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError>;

    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }

    /// Declare whether this integrator contributes a per-thread update
    /// to the JIT-composed post-force per-particle kernel. Returns
    /// `Some(self)` from an integrator that implements
    /// `PostForcePerParticle`, `None` (the default) otherwise. Every
    /// built-in integrator participates; a built-in returning `None` is
    /// the `StepError::MissingPostForcePerParticleFragment` rejection at
    /// runner construction. See
    /// `rqm/integration/jit-composed-post-force.md`.
    fn post_force_per_particle(&self) -> Option<&dyn PostForcePerParticle> {
        None
    }

    /// Returns the SubStep index in `plan(dt)` whose work is dispatched
    /// by the composed post-force kernel rather than by `execute`. The
    /// runner uses this to skip the integrator's per-step
    /// `execute(<that SubStep>, …)` call when the composed-kernel path
    /// is active. Default returns the index of the plan's last
    /// `KickHalf` or `KickDrift` SubStep, which matches the contract
    /// followed by every built-in integrator. Returns `None` if no
    /// such SubStep exists.
    fn post_force_substep_index(&self, dt: Real) -> Option<usize> {
        let plan = self.plan(dt);
        plan.steps.iter().enumerate().rev().find_map(|(idx, s)| {
            matches!(s, SubStep::KickHalf { .. } | SubStep::KickDrift { .. })
                .then_some(idx)
        })
    }

}

/// Per-call options for [`run_step`], bundling the four flags that
/// select among plan-walk modes. Plain `Copy` data; a caller overrides
/// individual fields against `Default`. See
/// `rqm/integration/framework.md`.
// rq-1d366b88
#[derive(Debug, Clone, Copy)]
pub struct RunStepOptions {
    /// `true` runs the neighbour-list pre-step via `force_field.step(...)`
    /// for each `ForceEval`; `false` calls
    /// `force_field.step_no_neighbor_check(...)` (CUDA-graph capture
    /// path). Default `true`.
    pub run_neighbor_pre_step: bool,
    /// `Some(i)` skips the integrator's `execute` for sub-step `i` (the
    /// JIT-composed post-force per-particle kernel handles it). Default
    /// `None`.
    pub skip_substep_index: Option<usize>,
    /// `true` (with a constraint slot passed) fires the constraint
    /// hooks at the canonical sub-step boundaries. Default `false`.
    pub install_constraint_hooks: bool,
    /// `true` resolves every `ForceEval` to
    /// `AggregateLevel::ForcesAndScalars`. Default `false`.
    pub runner_needs_scalars: bool,
}

impl Default for RunStepOptions {
    fn default() -> Self {
        RunStepOptions {
            run_neighbor_pre_step: true,
            skip_substep_index: None,
            install_constraint_hooks: false,
            runner_needs_scalars: false,
        }
    }
}

/// Walk an integrator's plan for one timestep — the single plan-walk
/// entry point.
///
/// Executes the integrator's sub-steps and the force pipeline together,
/// optionally weaving constraint-slot hook calls around any `Drift` or
/// `KickDrift` sub-step and after the final velocity update. The
/// per-step variations (graph-capture neighbour handling, composed
/// post-force skip, scalar-prep, constraint hooks) are selected by
/// `opts`; see [`RunStepOptions`].
///
/// `opts.install_constraint_hooks` should be `true` only when both
/// `constraint.is_some()` and the integrator's builder
/// `supports_constraints(&params)` would return `true`; otherwise no
/// hooks fire regardless of the `constraint` argument.
#[allow(clippy::too_many_arguments)]
pub fn run_step(
    integrator: &mut dyn Integrator,
    buffers: &mut ParticleBuffers,
    sim_box: &mut SimulationBox,
    force_field: &mut ForceField,
    mut constraint: Option<&mut dyn Constraint>,
    dt: Real,
    timings: &mut Timings,
    opts: RunStepOptions,
) -> Result<(), StepError> {
    let RunStepOptions {
        run_neighbor_pre_step,
        skip_substep_index,
        install_constraint_hooks,
        runner_needs_scalars,
    } = opts;
    let plan = integrator.plan(dt);
    let install = install_constraint_hooks && constraint.is_some();
    for (idx, sub) in plan.steps.iter().enumerate() {
        if Some(idx) == skip_substep_index {
            continue;
        }
        let is_drift = sub.is_drift();
        if install && is_drift {
            if let Some(c) = constraint.as_mut() {
                c.apply_before_drift(buffers, sim_box, dt, timings)?;
            }
        }
        match sub {
            SubStep::ForceEval { class: None, level } => {
                let resolved = resolve_aggregate_level(*level, runner_needs_scalars);
                if run_neighbor_pre_step {
                    force_field.step(buffers, sim_box, timings, resolved)?;
                } else {
                    force_field.step_no_neighbor_check(
                        buffers, sim_box, timings, resolved,
                    )?;
                }
            }
            SubStep::ForceEval {
                class: Some(c),
                level,
            } => {
                let resolved = resolve_aggregate_level(*level, runner_needs_scalars);
                if run_neighbor_pre_step {
                    force_field.step_class(*c, buffers, sim_box, timings, resolved)?;
                } else {
                    force_field.step_class_no_neighbor_check(
                        *c, buffers, sim_box, timings, resolved,
                    )?;
                }
            }
            other => {
                integrator.execute(other, buffers, sim_box, timings)?;
            }
        }
        if install && is_drift {
            if let Some(c) = constraint.as_mut() {
                c.apply_after_drift(buffers, sim_box, dt, timings)?;
            }
        }
    }
    let last_is_kick = plan
        .steps
        .last()
        .map(|s| s.is_velocity_update())
        .unwrap_or(false);
    if install && last_is_kick {
        if let Some(c) = constraint.as_mut() {
            c.apply_after_kick(buffers, sim_box, dt, timings)?;
        }
    }
    Ok(())
}

/// Walk an integrator's plan without any constraint-slot hooks.
/// Resolve the aggregation level for a single `SubStep::ForceEval`.
///
/// Returns `AggregateLevel::ForcesAndScalars` if either:
///   - the integrator's sub-step requested it explicitly
///     (`level == Some(ForcesAndScalars)`), or
///   - the runner independently requires scalars this step
///     (`runner_needs_scalars == true`; logging cadence, trajectory
///     cadence, minimization observation, etc.).
///
/// Otherwise returns the integrator's preference (defaulting to
/// `ForcesOnly` when the sub-step is `level: None`).
pub fn resolve_aggregate_level(
    sub_step_level: Option<crate::forces::AggregateLevel>,
    runner_needs_scalars: bool,
) -> crate::forces::AggregateLevel {
    use crate::forces::AggregateLevel;
    if runner_needs_scalars
        || matches!(sub_step_level, Some(AggregateLevel::ForcesAndScalars))
    {
        AggregateLevel::ForcesAndScalars
    } else {
        sub_step_level.unwrap_or(AggregateLevel::ForcesOnly)
    }
}


// rq-0e26dde0 rq-1ac78590
/// Extension trait offering a single-call `step()` convenience method
/// on top of the core `Integrator` trait's `plan()` + `execute()`
/// methods. The trait itself defines only the plan/execute pair (see
/// `framework.md`); this extension is purely a convenience wrapper for
/// callers — chiefly tests — that want a single method invocation per
/// timestep. The runner uses the lower-level [`run_step`] free
/// function directly.
///
/// `step` walks the plan with no constraint slot. Callers that need
/// a constraint slot installed use [`IntegratorStepWithConstraintExt::step_with_constraint`],
/// which is bounded on `Self: ConstraintCapableIntegrator` and is
/// therefore unavailable on integrators whose plan shape is not
/// constraint-compatible.
pub trait IntegratorStepExt {
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), StepError>;
}

impl IntegratorStepExt for dyn Integrator + '_ {
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), StepError> {
        run_step(
            self,
            buffers,
            sim_box,
            force_field,
            None,
            dt,
            timings,
            RunStepOptions { runner_needs_scalars: true, ..Default::default() },
        )
    }
}

// Blanket impl for concrete (Sized) integrators so tests can call
// `state.step(...)` directly without coercing to `&mut dyn Integrator`.
impl<T: Integrator> IntegratorStepExt for T {
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), StepError> {
        run_step(
            self,
            buffers,
            sim_box,
            force_field,
            None,
            dt,
            timings,
            RunStepOptions { runner_needs_scalars: true, ..Default::default() },
        )
    }
}

// rq-0e26dde0 rq-ab8c77bc
/// Marker (with a runtime-state predicate) for integrator types whose
/// `StepPlan` shape is compatible with the constraint slot's hook
/// positions. Implemented by integrators whose plans place a single
/// `Drift` / `KickDrift` sub-step and a terminal `KickHalf` /
/// `KickDrift` — currently `VelocityVerletState`.
///
/// `check_accepts_constraints_now` is consulted at runtime by
/// `IntegratorStepWithConstraintExt::step_with_constraint`. The
/// default returns `Ok(())`; implementations whose internal state can
/// transiently forbid hook installation (e.g., `VelocityVerletState`
/// with `lossless = true`) override it to return `Err(reason)`. The
/// returned message propagates verbatim into
/// `StepError::IntegratorRejectsConstraint { reason }`.
pub trait ConstraintCapableIntegrator: Integrator {
    fn check_accepts_constraints_now(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

// rq-0e26dde0 rq-f71ff87f
/// Extension trait offering a single-call `step_with_constraint()`
/// convenience method on top of [`IntegratorStepExt::step`]. Bounded
/// on `Self: ConstraintCapableIntegrator`, so only the integrator
/// types whose plan shape is constraint-compatible expose the method
/// at all. Calling `step_with_constraint` on `LangevinBaoabState`,
/// `MtkNptIntegrator`, or any other non-marker type is a compile
/// error.
///
/// Before walking the plan, the method calls
/// `self.check_accepts_constraints_now()`. If it returns
/// `Err(reason)`, the method returns
/// `Err(StepError::IntegratorRejectsConstraint { reason })` without
/// dispatching `plan()`, `execute()`, the force field, or any
/// constraint hook.
pub trait IntegratorStepWithConstraintExt {
    fn step_with_constraint(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        constraint: &mut dyn Constraint,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), StepError>;
}

impl IntegratorStepWithConstraintExt for dyn ConstraintCapableIntegrator + '_ {
    fn step_with_constraint(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        constraint: &mut dyn Constraint,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), StepError> {
        if let Err(reason) = self.check_accepts_constraints_now() {
            return Err(StepError::IntegratorRejectsConstraint { reason });
        }
        run_step(
            self,
            buffers,
            sim_box,
            force_field,
            Some(constraint),
            dt,
            timings,
            RunStepOptions {
                install_constraint_hooks: true,
                runner_needs_scalars: true,
                ..Default::default()
            },
        )
    }
}

// Blanket impl for concrete (Sized) integrators so tests can call
// `state.step_with_constraint(...)` directly.
impl<T: ConstraintCapableIntegrator> IntegratorStepWithConstraintExt for T {
    fn step_with_constraint(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        constraint: &mut dyn Constraint,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), StepError> {
        if let Err(reason) = self.check_accepts_constraints_now() {
            return Err(StepError::IntegratorRejectsConstraint { reason });
        }
        run_step(
            self,
            buffers,
            sim_box,
            force_field,
            Some(constraint),
            dt,
            timings,
            RunStepOptions {
                install_constraint_hooks: true,
                runner_needs_scalars: true,
                ..Default::default()
            },
        )
    }
}

// rq-29e08cb5
pub trait IntegratorBuilder:
    KindedBuilder + IntegratorBuilderClone + std::fmt::Debug + Send + Sync
{
    /// Validate the kind-specific parameters of an `[integrator]`
    /// section at config-load time. Implementations deserialise the
    /// `toml::Value` into their typed parameter struct and surface
    /// every domain check as a `ConfigError::InvalidValue` (or one
    /// of the more specific `ConfigError` variants).
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>;

    /// `true` iff the integrator fuses its own thermostat (so
    /// composing it with a `[thermostat]` slot is rejected at load
    /// time). The default returns `false`.
    fn owns_thermostat(&self, _params: &toml::Value) -> bool {
        false
    }

    /// `true` iff the integrator fuses its own barostat. Default
    /// `false`.
    fn owns_barostat(&self, _params: &toml::Value) -> bool {
        false
    }

    /// `true` iff the integrator drives the three `Constraint` slot
    /// hooks (see `constraint-framework.md`). Default `false`.
    fn supports_constraints(&self, _params: &toml::Value) -> bool {
        false
    }

    /// `true` iff every per-step entry point (`step`, `execute`, and
    /// any sub-step surfaced through `Plan`) consists of pure CUDA
    /// kernel launches with no host-side state mutation between
    /// launches and no `dtoh_sync_copy` / `htod_sync_copy` calls.
    /// Determines whether phases driven by this integrator run under
    /// CUDA graph mode. Default `true`; integrators with host-side
    /// scalar arithmetic inside the plan executor override to
    /// `false`.
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        true
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError>;
}

// rq-4901507f
pub type IntegratorRegistry = Registry<dyn IntegratorBuilder>;

impl Builtins for dyn IntegratorBuilder {
    fn builtins() -> Vec<Box<dyn IntegratorBuilder>> {
        vec![
            Box::new(VelocityVerletBuilder),
            Box::new(LangevinBaoabBuilder),
            Box::new(MtkNptBuilder),
        ]
    }
}

crate::registry_builder_clone!(pub IntegratorBuilderClone for IntegratorBuilder);

impl Registry<dyn IntegratorBuilder> {
    // rq-24f6b8b9 rq-1e30bbf4
    pub fn build(
        &self,
        slot: &SlotConfig,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let b = self
            .lookup(&slot.kind)
            .ok_or_else(|| IntegratorError::UnknownKind(slot.kind.clone()))?;
        b.build(gpu, particle_count, n_constraints, &slot.params)
    }
}

// --- Thermostat trait, builder, registry ------------------------------

// rq-5d9ed248
pub trait Thermostat: std::fmt::Debug + Send {
    // rq-2fe47a86
    fn apply_pre(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _dt: Real,
        _timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        Ok(())
    }

    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError>;

    /// Drain any device-side accumulators the thermostat maintains
    /// (e.g. CSVR's `(k_new - k_old)` delta) into host state so that
    /// `log_column_values` reflects every step since the last flush.
    /// Default implementation is a no-op for thermostats that maintain
    /// no device-side accumulator. The runner calls this once before
    /// each log row is emitted.
    fn flush_pending_injection(
        &mut self,
        _device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), ThermostatError> {
        Ok(())
    }

    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }

    /// Declare whether this thermostat contributes a per-thread rescale
    /// / resample to the JIT-composed post-force per-particle kernel.
    /// Returns `Some(self)` from a thermostat that implements
    /// `PostForcePerParticle`, `None` (the default) otherwise. Built-in
    /// thermostats participate. See
    /// `rqm/integration/jit-composed-post-force.md`.
    fn post_force_per_particle(&self) -> Option<&dyn PostForcePerParticle> {
        None
    }

}

// rq-29e08cb5
pub trait ThermostatBuilder:
    KindedBuilder + ThermostatBuilderClone + std::fmt::Debug + Send + Sync
{
    /// Validate the kind-specific parameters of a `[thermostat]`
    /// section at config-load time.
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>;

    /// `true` iff every thermostat entry point (`apply_pre`,
    /// `apply_post`) consists of pure CUDA kernel launches with no
    /// host-side state mutation between launches. Determines whether
    /// phases using this thermostat run under CUDA graph mode. Default
    /// `true`.
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        true
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError>;
}

// rq-4901507f
pub type ThermostatRegistry = Registry<dyn ThermostatBuilder>;

impl Builtins for dyn ThermostatBuilder {
    fn builtins() -> Vec<Box<dyn ThermostatBuilder>> {
        vec![
            Box::new(NoseHooverChainBuilder),
            Box::new(CsvrBuilder),
            Box::new(AndersenBuilder),
            Box::new(BerendsenBuilder),
        ]
    }
}

crate::registry_builder_clone!(pub ThermostatBuilderClone for ThermostatBuilder);

impl Registry<dyn ThermostatBuilder> {
    // rq-678c233d
    pub fn build_optional(
        &self,
        slot: Option<&SlotConfig>,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
    ) -> Result<Option<Box<dyn Thermostat>>, ThermostatError> {
        let Some(slot) = slot else { return Ok(None) };
        let b = self
            .lookup(&slot.kind)
            .ok_or_else(|| ThermostatError::UnknownKind(slot.kind.clone()))?;
        Ok(Some(b.build(gpu, particle_count, n_constraints, &slot.params)?))
    }
}

// --- Barostat trait, builder, registry --------------------------------

// rq-076617ab
pub trait Barostat: std::fmt::Debug + Send {
    // rq-1179e42f
    fn apply(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), BarostatError>;

    /// Drain any device-side accumulators the barostat maintains
    /// (e.g. C-rescale's `P_target · (v_post - v_pre)` delta) into host
    /// state so that `log_column_values` reflects every step since the
    /// last flush. Default implementation is a no-op for barostats that
    /// maintain no device-side accumulator. The runner calls this once
    /// before each log row is emitted.
    fn flush_pending_injection(
        &mut self,
        _device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), BarostatError> {
        Ok(())
    }

    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }

    /// Declare whether this barostat contributes a per-thread rescale
    /// to the JIT-composed post-force per-particle kernel. Returns
    /// `Some(self)` from a barostat that implements
    /// `PostForcePerParticle`, `None` (the default) otherwise. Built-in
    /// barostats participate. See
    /// `rqm/integration/jit-composed-post-force.md`.
    fn post_force_per_particle(&self) -> Option<&dyn PostForcePerParticle> {
        None
    }

}

// rq-29e08cb5
pub trait BarostatBuilder:
    KindedBuilder + BarostatBuilderClone + std::fmt::Debug + Send + Sync
{
    /// Validate the kind-specific parameters of a `[barostat]`
    /// section at config-load time.
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>;

    /// `true` iff `Barostat::apply` consists of pure CUDA kernel
    /// launches with no host-side state mutation between launches.
    /// Determines whether phases using this barostat run under CUDA
    /// graph mode. Default `true`.
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        true
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Barostat>, BarostatError>;
}

// rq-4901507f
pub type BarostatRegistry = Registry<dyn BarostatBuilder>;

impl Builtins for dyn BarostatBuilder {
    fn builtins() -> Vec<Box<dyn BarostatBuilder>> {
        vec![
            Box::new(BerendsenBarostatBuilder),
            Box::new(CRescaleBarostatBuilder),
        ]
    }
}

crate::registry_builder_clone!(pub BarostatBuilderClone for BarostatBuilder);

impl Registry<dyn BarostatBuilder> {
    // rq-9548bc1a
    pub fn build_optional(
        &self,
        slot: Option<&SlotConfig>,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
    ) -> Result<Option<Box<dyn Barostat>>, BarostatError> {
        let Some(slot) = slot else { return Ok(None) };
        let b = self
            .lookup(&slot.kind)
            .ok_or_else(|| BarostatError::UnknownKind(slot.kind.clone()))?;
        Ok(Some(b.build(gpu, particle_count, n_constraints, &slot.params)?))
    }
}

