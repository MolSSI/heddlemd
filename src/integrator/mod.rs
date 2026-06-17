// rq-e0a0553d rq-6cd635cd rq-6c5b4246
//
// Three orthogonal slot frameworks: integrator, thermostat, barostat.
// The runner chains the slots `apply_pre → step → apply_post → apply`
// per timestep (see `simulation-runner.md` and `framework.md`).

use crate::forces::{ForceField, ForceFieldError};
use crate::gpu::{GpuContext, GpuError, ParticleBuffers};
use crate::io::config::{ConfigError, SlotConfig};
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
}

/// Walk an integrator's plan for one timestep.
///
/// The runner uses this to execute integrator sub-steps and the
/// force pipeline together, optionally weaving constraint-slot hook
/// calls around any `Drift` or `KickDrift` sub-step and after the
/// final velocity update.
///
/// `install_constraint_hooks` must be `true` only when both
/// `constraint.is_some()` and the integrator's
/// `IntegratorKind::supports_constraints()` predicate would return
/// `true`. When `false`, no constraint hooks fire regardless of the
/// `constraint` argument.
#[allow(clippy::too_many_arguments)]
pub fn run_step(
    integrator: &mut dyn Integrator,
    buffers: &mut ParticleBuffers,
    sim_box: &mut SimulationBox,
    force_field: &mut ForceField,
    mut constraint: Option<&mut dyn Constraint>,
    install_constraint_hooks: bool,
    dt: Real,
    timings: &mut Timings,
    runner_needs_scalars: bool,
) -> Result<(), StepError> {
    let plan = integrator.plan(dt);
    let install = install_constraint_hooks && constraint.is_some();
    for sub in &plan.steps {
        let is_drift = sub.is_drift();
        if install && is_drift {
            if let Some(c) = constraint.as_mut() {
                c.apply_before_drift(buffers, sim_box, dt, timings)?;
            }
        }
        match sub {
            SubStep::ForceEval { class: None, level } => {
                let resolved = resolve_aggregate_level(*level, runner_needs_scalars);
                force_field.step(buffers, sim_box, timings, resolved)?;
            }
            SubStep::ForceEval {
                class: Some(c),
                level,
            } => {
                let resolved = resolve_aggregate_level(*level, runner_needs_scalars);
                force_field.step_class(*c, buffers, sim_box, timings, resolved)?;
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

/// Convenience for tests and callers that don't need constraints.
/// Always requests `ForcesAndScalars` so tests that read energy / virial
/// after `run_step_no_constraint` see fresh values regardless of the
/// integrator's per-step level preference.
pub fn run_step_no_constraint(
    integrator: &mut dyn Integrator,
    buffers: &mut ParticleBuffers,
    sim_box: &mut SimulationBox,
    force_field: &mut ForceField,
    dt: Real,
    timings: &mut Timings,
) -> Result<(), StepError> {
    run_step(
        integrator,
        buffers,
        sim_box,
        force_field,
        None,
        false,
        dt,
        timings,
        true,
    )
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
        run_step(self, buffers, sim_box, force_field, None, false, dt, timings, true)
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
        run_step(self, buffers, sim_box, force_field, None, false, dt, timings, true)
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
            true,
            dt,
            timings,
            true,
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
            true,
            dt,
            timings,
            true,
        )
    }
}

// rq-29e08cb5
pub trait IntegratorBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;

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

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError>;

    /// Return a clone of `self` boxed as a trait object. Used to
    /// implement `Clone` for `IntegratorRegistry` (which holds
    /// `Vec<Box<dyn IntegratorBuilder>>`).
    fn box_clone(&self) -> Box<dyn IntegratorBuilder>;
}

// rq-4901507f
#[derive(Debug)]
pub struct IntegratorRegistry {
    pub builders: Vec<Box<dyn IntegratorBuilder>>,
}

impl Clone for IntegratorRegistry {
    fn clone(&self) -> Self {
        IntegratorRegistry {
            builders: self.builders.iter().map(|b| b.box_clone()).collect(),
        }
    }
}

impl IntegratorRegistry {
    pub fn new() -> Self {
        IntegratorRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        IntegratorRegistry {
            builders: vec![
                Box::new(VelocityVerletBuilder),
                Box::new(LangevinBaoabBuilder),
                Box::new(MtkNptBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn IntegratorBuilder>) {
        self.builders.push(builder);
    }

    /// Return the first registered builder whose `kind_name()` equals
    /// `kind`. The runner uses this both to query compatibility
    /// predicates and to drive `validate_params` at config-load time.
    pub fn lookup(&self, kind: &str) -> Option<&dyn IntegratorBuilder> {
        for b in &self.builders {
            if b.kind_name() == kind {
                return Some(b.as_ref());
            }
        }
        None
    }

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

impl Default for IntegratorRegistry {
    fn default() -> Self {
        IntegratorRegistry::with_builtins()
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
}

// rq-29e08cb5
pub trait ThermostatBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;

    /// Validate the kind-specific parameters of a `[thermostat]`
    /// section at config-load time.
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>;

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError>;

    fn box_clone(&self) -> Box<dyn ThermostatBuilder>;
}

// rq-4901507f
#[derive(Debug)]
pub struct ThermostatRegistry {
    pub builders: Vec<Box<dyn ThermostatBuilder>>,
}

impl Clone for ThermostatRegistry {
    fn clone(&self) -> Self {
        ThermostatRegistry {
            builders: self.builders.iter().map(|b| b.box_clone()).collect(),
        }
    }
}

impl ThermostatRegistry {
    pub fn new() -> Self {
        ThermostatRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        ThermostatRegistry {
            builders: vec![
                Box::new(NoseHooverChainBuilder),
                Box::new(CsvrBuilder),
                Box::new(AndersenBuilder),
                Box::new(BerendsenBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn ThermostatBuilder>) {
        self.builders.push(builder);
    }

    // rq-c44b25af
    pub fn lookup(&self, kind: &str) -> Option<&dyn ThermostatBuilder> {
        for b in &self.builders {
            if b.kind_name() == kind {
                return Some(b.as_ref());
            }
        }
        None
    }

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

impl Default for ThermostatRegistry {
    fn default() -> Self {
        ThermostatRegistry::with_builtins()
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
}

// rq-29e08cb5
pub trait BarostatBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;

    /// Validate the kind-specific parameters of a `[barostat]`
    /// section at config-load time.
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>;

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Barostat>, BarostatError>;

    fn box_clone(&self) -> Box<dyn BarostatBuilder>;
}

// rq-4901507f
#[derive(Debug)]
pub struct BarostatRegistry {
    pub builders: Vec<Box<dyn BarostatBuilder>>,
}

impl Clone for BarostatRegistry {
    fn clone(&self) -> Self {
        BarostatRegistry {
            builders: self.builders.iter().map(|b| b.box_clone()).collect(),
        }
    }
}

impl BarostatRegistry {
    pub fn new() -> Self {
        BarostatRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        BarostatRegistry {
            builders: vec![
                Box::new(BerendsenBarostatBuilder),
                Box::new(CRescaleBarostatBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn BarostatBuilder>) {
        self.builders.push(builder);
    }

    // rq-acbb6d0e
    pub fn lookup(&self, kind: &str) -> Option<&dyn BarostatBuilder> {
        for b in &self.builders {
            if b.kind_name() == kind {
                return Some(b.as_ref());
            }
        }
        None
    }

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

impl Default for BarostatRegistry {
    fn default() -> Self {
        BarostatRegistry::with_builtins()
    }
}

