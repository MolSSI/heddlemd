// rq-e0a0553d rq-6cd635cd rq-6c5b4246
//
// Three orthogonal slot frameworks: integrator, thermostat, barostat.
// The runner chains the slots `apply_pre → step → apply_post → apply`
// per timestep (see `simulation-runner.md` and `framework.md`).

use crate::forces::{ForceField, ForceFieldError, MoleculeList};
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
pub mod mc_barostat;
pub mod mtk_npt;
pub mod nose_hoover_chain;
pub mod philox;
pub mod respa;
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
pub use mc_barostat::{McBarostat, McBarostatBuilder};
pub use mtk_npt::{MtkNptBuilder, MtkNptIntegrator};
pub use nose_hoover_chain::{
    NoseHooverChainBuilder, NoseHooverChainThermostat, nhc_chain_sub_step,
};
pub use philox::{philox_4x32_10, philox_normal};
pub use respa::{RespaBuilder, RespaIntegrator};
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
    /// A runner-dispatched class-sourced kick launch failed.
    #[error("{0}")]
    Gpu(#[from] GpuError),
    /// A plan-declared `ThermostatHalf` dispatch failed.
    #[error("{0}")]
    Thermostat(#[from] ThermostatError),
    /// A plan-declared `BarostatPoint` dispatch failed.
    // rq-dbbffa7d
    #[error("{0}")]
    Barostat(#[from] BarostatError),
    #[error("{0}")]
    Timings(#[from] crate::timings::TimingsError),
    // rq-0e26dde0
    /// Returned by `IntegratorStepWithConstraintExt::step_with_constraint`
    /// when the integrator's `check_accepts_constraints_now()`
    /// rejected hook installation for the instance's current runtime
    /// state (e.g., velocity-Verlet with `lossless = true`). The
    /// `reason` is the integrator's verbatim message.
    #[error("integrator rejected constraint hook installation: {reason}")]
    IntegratorRejectsConstraint { reason: &'static str },
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
    /// A force-field evaluation issued by a barostat move (the
    /// Monte-Carlo barostat's trial energy evaluation) failed.
    #[error("{0}")]
    ForceField(#[from] ForceFieldError),
    /// A simulation-box mutation issued by a barostat move failed.
    #[error("{0}")]
    SimulationBox(#[from] crate::pbc::SimulationBoxError),
    #[error("unknown barostat kind `{0}`")]
    UnknownKind(String),
}

// --- Integrator trait, builder, registry ------------------------------

// rq-dbbffa7d
/// Selects the force a `KickHalf` / `KickDrift` consumes, and thereby
/// its dispatcher: `Total` kicks read the combined
/// `ParticleBuffers.forces_*` and are executed by the integrator;
/// `Class` kicks read one class accumulator and are dispatched by the
/// runner (only the runner holds the `ForceField`), via the
/// framework-owned `class_kick_half` / `class_kick_drift` kernels.
/// Class-sourced kicks are the impulse-splitting form used by RESPA
/// (`rqm/integration/respa.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KickSource {
    Total,
    Class(crate::forces::ForceClass),
}

// rq-dbbffa7d
/// Which thermostat half a `SubStep::ThermostatHalf` dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermostatPhase {
    Pre,
    Post,
}

// rq-eea8aa89
/// Which constraint hook a `SubStep::ConstraintPoint` dispatches. Each
/// variant maps to one `Constraint` trait method (see
/// `rqm/integration/constraint-framework.md`). A constraint-capable
/// integrator places these markers in its plan to declare where the
/// slot must snapshot or project; the runner does no structural
/// inference from the plan's kick / drift shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintPhase {
    /// `constraint.apply_before_drift(...)` — snapshot pre-drift
    /// positions. Placed immediately before a position-updating
    /// sub-step.
    BeforeDrift,
    /// `constraint.apply_after_drift(...)` — project positions onto the
    /// constraint manifold and update the corresponding half-step
    /// velocities. Placed immediately after a position-updating
    /// sub-step.
    AfterDrift,
    /// `constraint.apply_after_kick(...)` — project velocities onto the
    /// constraint manifold. Placed after a velocity-updating sub-step.
    /// When the composed post-force kernel absorbs the preceding kick,
    /// a plan-final `AfterKick` marker is displaced past the composed
    /// launch and fired by the runner (see `framework.md`).
    AfterKick,
}

// rq-dbbffa7d
/// One piece of an integrator's per-timestep work, described in the
/// `StepPlan` returned by [`Integrator::plan`].
#[derive(Debug, Clone, Copy)]
pub enum SubStep {
    /// Velocity half-kick: `v ← v + (F/m) · dt/2` (or the integrator's
    /// equivalent). No position update. `source` selects the consumed
    /// force and the dispatcher (see [`KickSource`]).
    KickHalf { dt: Real, label: &'static str, source: KickSource },
    /// Position drift: `x ← x + v · dt` (or the integrator's
    /// equivalent). No velocity update.
    Drift { dt: Real, label: &'static str },
    /// Fused KickHalf + Drift in a single kernel launch
    /// (e.g. `vv_kick_drift`): the kick part uses `dt/2`, the drift
    /// part `dt`. `source` as on `KickHalf`.
    KickDrift { dt: Real, label: &'static str, source: KickSource },
    /// Dispatch the configured thermostat's pre- or post-half here
    /// with the given `dt`. Dispatched by the runner, not by
    /// `execute()`; a no-op when the run has no thermostat. A plan
    /// containing any `ThermostatHalf` owns its thermostat placement:
    /// the runner's default apply_pre/apply_post wrapping is
    /// suppressed for that plan.
    ThermostatHalf { dt: Real, phase: ThermostatPhase },
    /// Dispatch the configured constraint slot's hook here with the
    /// given `dt`. Dispatched by the runner, not by `execute()`; a
    /// no-op when the run has no constraint slot. `phase` selects the
    /// hook (see [`ConstraintPhase`]). `dt` is the sub-step timestep
    /// the projection operates over, so its velocity and virial factors
    /// use the correct interval.
    // rq-dbbffa7d
    ConstraintPoint { phase: ConstraintPhase, dt: Real },
    /// Dispatch the configured per-step barostat's `apply` here with the
    /// given `dt`. Dispatched by the runner, not by `execute()`; a no-op
    /// when no per-step barostat is configured (and inert for a periodic
    /// barostat, whose `apply` is the no-op default). A terminal
    /// `BarostatPoint` is the canonical placement (the runner fuses the
    /// per-particle rescale into the composed post-force kernel); a
    /// mid-plan `BarostatPoint` runs `apply` standalone during the walk.
    // rq-dbbffa7d
    BarostatPoint { dt: Real },
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
            SubStep::ThermostatHalf { .. } => "ThermostatHalf",
            SubStep::ConstraintPoint { .. } => "ConstraintPoint",
            SubStep::BarostatPoint { .. } => "BarostatPoint",
            SubStep::ForceEval { .. } => "ForceEval",
            SubStep::Custom { .. } => "Custom",
        }
    }

    // rq-dbbffa7d — `true` iff this sub-step is a trailing post-force
    // marker: a terminal constraint velocity projection or a barostat
    // point. The runner dispatches the trailing run of these in its
    // post-walk tail rather than the plan walk.
    fn is_post_force_marker(&self) -> bool {
        matches!(
            self,
            SubStep::ConstraintPoint {
                phase: ConstraintPhase::AfterKick,
                ..
            } | SubStep::BarostatPoint { .. }
        )
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

    // rq-9fbba3be
    /// `true` iff any sub-step is a `ThermostatHalf`. The runner
    /// consults this to choose the thermostat topology (marker-bearing
    /// plans own their thermostat placement and receive no default
    /// apply_pre/apply_post wrapping) and to exclude marker-bearing
    /// plans from CUDA-graph capture.
    pub fn has_thermostat_points(&self) -> bool {
        self.steps
            .iter()
            .any(|s| matches!(s, SubStep::ThermostatHalf { .. }))
    }

    // rq-9fbba3be
    /// Index at which the plan's **trailing post-force markers** begin —
    /// the maximal trailing run of `ConstraintPoint { AfterKick }` and
    /// `BarostatPoint` sub-steps. Equals `steps.len()` when the final
    /// sub-step is not a post-force marker. The runner dispatches this
    /// trailing run in its post-walk tail so it composes with the
    /// composed post-force kernel (see `framework.md`).
    pub fn trailing_post_force_start(&self) -> usize {
        let mut start = self.steps.len();
        while start > 0 && self.steps[start - 1].is_post_force_marker() {
            start -= 1;
        }
        start
    }

    // rq-9fbba3be
    /// `true` iff the trailing post-force-marker run contains a
    /// `ConstraintPoint { phase: AfterKick }`. The runner consults this
    /// to decide whether a trailing velocity projection is displaced
    /// past the composed post-force launch (see `framework.md`).
    /// `ConstraintPoint` markers dispatch pure kernel launches and never
    /// disqualify a plan from CUDA-graph capture, so — unlike
    /// `has_thermostat_points()` — this predicate does not affect graph
    /// eligibility.
    pub fn ends_with_velocity_projection(&self) -> bool {
        self.terminal_velocity_projection_dt().is_some()
    }

    // rq-9fbba3be
    /// The `dt` carried by the `ConstraintPoint { phase: AfterKick }` in
    /// the trailing post-force-marker run, or `None` when there is none.
    /// The runner passes the `Some` value to the displaced
    /// `apply_after_kick` fired after the composed launch.
    pub fn terminal_velocity_projection_dt(&self) -> Option<Real> {
        self.steps[self.trailing_post_force_start()..]
            .iter()
            .find_map(|s| match s {
                SubStep::ConstraintPoint {
                    phase: ConstraintPhase::AfterKick,
                    dt,
                } => Some(*dt),
                _ => None,
            })
    }

    // rq-9fbba3be
    /// `true` iff any sub-step is a `BarostatPoint`.
    pub fn has_barostat_points(&self) -> bool {
        self.steps
            .iter()
            .any(|s| matches!(s, SubStep::BarostatPoint { .. }))
    }

    // rq-9fbba3be
    /// The `dt` carried by the `BarostatPoint` in the trailing
    /// post-force-marker run (the canonical, composed-fused placement),
    /// or `None` when the plan has no terminal `BarostatPoint`. A
    /// `BarostatPoint` outside the trailing run (interleaved) is not
    /// reported here.
    pub fn terminal_barostat_point_dt(&self) -> Option<Real> {
        self.steps[self.trailing_post_force_start()..]
            .iter()
            .find_map(|s| match s {
                SubStep::BarostatPoint { dt } => Some(*dt),
                _ => None,
            })
    }

    // rq-9fbba3be
    /// `true` iff the plan carries a `BarostatPoint` outside the trailing
    /// post-force-marker run (interleaved placement). Such a plan runs on
    /// the eager path: the mid-walk barostat arithmetic cannot be
    /// captured into a CUDA graph.
    pub fn has_interleaved_barostat_point(&self) -> bool {
        let start = self.trailing_post_force_start();
        self.steps[..start]
            .iter()
            .any(|s| matches!(s, SubStep::BarostatPoint { .. }))
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
    /// `PostForcePerParticle`, `None` (the default) otherwise.
    /// Participation is optional: a non-participating integrator's
    /// trailing kick executes in the plan walk (the runner derives
    /// the skip index from participation, so the kick is never
    /// silently lost). Every built-in integrator participates —
    /// enforced by a registry lint test. See
    /// `rqm/integration/jit-composed-post-force.md` (rq-09306735).
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
    /// `true` makes `run_step` skip dispatch of the plan's final
    /// sub-step when it is a `ConstraintPoint { phase: AfterKick }`; the
    /// runner fires the corresponding `apply_after_kick` after the
    /// composed post-force launch so the projection follows the fused
    /// kick. Set by the runner exactly when the composed kernel absorbs
    /// the integrator's trailing kick and a constraint slot is
    /// installed. Default `false`.
    pub defer_terminal_velocity_projection: bool,
    /// `true` makes `run_step` skip dispatch of the trailing-post-force-
    /// marker `BarostatPoint`; the runner fires `barostat.apply` in the
    /// post-walk tail (before the composed launch) so the barostat's
    /// per-particle rescale fuses into the composed kernel. Set by the
    /// runner when a per-step barostat is configured and the plan carries
    /// a terminal `BarostatPoint`. Default `false` dispatches every
    /// `BarostatPoint` in the walk as a full standalone `apply`.
    pub defer_terminal_barostat_point: bool,
    /// `true` resolves every `ForceEval` to
    /// `AggregateLevel::ForcesAndScalars`. Default `false`.
    pub runner_needs_scalars: bool,
}

impl Default for RunStepOptions {
    fn default() -> Self {
        RunStepOptions {
            run_neighbor_pre_step: true,
            skip_substep_index: None,
            defer_terminal_velocity_projection: false,
            defer_terminal_barostat_point: false,
            runner_needs_scalars: false,
        }
    }
}

/// Walk an integrator's plan for one timestep — the single plan-walk
/// entry point.
///
/// Executes the integrator's sub-steps and the force pipeline together.
/// `ConstraintPoint` markers dispatch to the constraint slot (a no-op
/// when `constraint` is `None`); the other per-step variations
/// (graph-capture neighbour handling, composed post-force skip,
/// deferred terminal projection, scalar-prep) are selected by `opts`;
/// see [`RunStepOptions`].
///
/// When `opts.defer_terminal_velocity_projection` is `true`, the plan's
/// final sub-step is skipped if it is a
/// `ConstraintPoint { phase: AfterKick }`; the runner fires the
/// corresponding `apply_after_kick` after the composed post-force launch
/// so the projection follows the composed kernel's fused kick.
#[allow(clippy::too_many_arguments)]
pub fn run_step(
    integrator: &mut dyn Integrator,
    buffers: &mut ParticleBuffers,
    sim_box: &mut SimulationBox,
    force_field: &mut ForceField,
    mut constraint: Option<&mut dyn Constraint>,
    mut thermostat: Option<&mut dyn Thermostat>,
    mut barostat: Option<&mut dyn Barostat>,
    dt: Real,
    timings: &mut Timings,
    opts: RunStepOptions,
) -> Result<(), StepError> {
    let RunStepOptions {
        run_neighbor_pre_step,
        skip_substep_index,
        defer_terminal_velocity_projection,
        defer_terminal_barostat_point,
        runner_needs_scalars,
    } = opts;
    let plan = integrator.plan(dt);
    // rq-277dbeb2 — the trailing run of post-force markers (terminal
    // velocity projection and/or terminal barostat point) that the
    // runner dispatches in its post-walk tail rather than the walk.
    let trailing_start = plan.trailing_post_force_start();
    for (idx, sub) in plan.steps.iter().enumerate() {
        if Some(idx) == skip_substep_index {
            continue;
        }
        // rq-277dbeb2 — the runner fires the trailing velocity
        // projection after the composed post-force launch (which
        // performs the absorbed kick), so skip it here.
        if defer_terminal_velocity_projection
            && idx >= trailing_start
            && matches!(
                sub,
                SubStep::ConstraintPoint {
                    phase: ConstraintPhase::AfterKick,
                    ..
                }
            )
        {
            continue;
        }
        // rq-dbbffa7d — the runner fires the trailing barostat point in
        // the post-walk tail (its per-particle rescale fuses into the
        // composed launch), so skip it here.
        if defer_terminal_barostat_point
            && idx >= trailing_start
            && matches!(sub, SubStep::BarostatPoint { .. })
        {
            continue;
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
            // rq-277dbeb2 — class-sourced kicks are runner-dispatched:
            // only the runner holds the ForceField and its per-class
            // accumulators.
            SubStep::KickHalf {
                dt: sub_dt,
                source: KickSource::Class(c),
                ..
            } => {
                timings.kernel_start(crate::timings::KernelStage::CLASS_KICK_HALF)?;
                crate::gpu::class_kick_half(buffers, force_field.class_forces(*c), *sub_dt)?;
                timings.kernel_stop(crate::timings::KernelStage::CLASS_KICK_HALF)?;
            }
            SubStep::KickDrift {
                dt: sub_dt,
                source: KickSource::Class(c),
                ..
            } => {
                timings.kernel_start(crate::timings::KernelStage::CLASS_KICK_DRIFT)?;
                crate::gpu::class_kick_drift(
                    buffers,
                    force_field.class_forces(*c),
                    sim_box,
                    *sub_dt,
                )?;
                timings.kernel_stop(crate::timings::KernelStage::CLASS_KICK_DRIFT)?;
            }
            // rq-277dbeb2 — plan-declared thermostat point; a no-op
            // when the run has no thermostat.
            SubStep::ThermostatHalf { dt: sub_dt, phase } => {
                if let Some(t) = thermostat.as_mut() {
                    match phase {
                        ThermostatPhase::Pre => t.apply_pre(buffers, *sub_dt, timings)?,
                        ThermostatPhase::Post => t.apply_post(buffers, *sub_dt, timings)?,
                    }
                }
            }
            // rq-dbbffa7d — plan-declared constraint hook; a no-op when
            // the run has no constraint slot. Each phase dispatches to
            // the matching `Constraint` trait method with the marker's
            // own `dt`.
            SubStep::ConstraintPoint { phase, dt: sub_dt } => {
                if let Some(c) = constraint.as_mut() {
                    match phase {
                        ConstraintPhase::BeforeDrift => {
                            c.apply_before_drift(buffers, sim_box, *sub_dt, timings)?
                        }
                        ConstraintPhase::AfterDrift => {
                            c.apply_after_drift(buffers, sim_box, *sub_dt, timings)?
                        }
                        ConstraintPhase::AfterKick => {
                            c.apply_after_kick(buffers, sim_box, *sub_dt, timings)?
                        }
                    }
                }
            }
            // rq-dbbffa7d — plan-declared, interleaved (non-terminal)
            // barostat point: a full standalone `apply` during the walk.
            // A no-op when no per-step barostat is configured (and inert
            // for a periodic barostat, whose `apply` is the no-op
            // default). Terminal barostat points are deferred above and
            // fired by the runner's post-walk tail.
            SubStep::BarostatPoint { dt: sub_dt } => {
                if let Some(b) = barostat.as_mut() {
                    b.apply(buffers, sim_box, *sub_dt, timings)?;
                }
            }
            other => {
                integrator.execute(other, buffers, sim_box, timings)?;
            }
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
            None,
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
            None,
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
            None,
            None,
            dt,
            timings,
            RunStepOptions {
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
            None,
            None,
            dt,
            timings,
            RunStepOptions {
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

    /// `true` iff the integrator may be combined with an external
    /// `[barostat]` slot. Default `true`; RESPA returns `false`
    /// (RESPA-NPT splittings are a separate design — see
    /// `rqm/integration/respa.md`). Distinct from `owns_barostat`,
    /// which signals a *fused* barostat. rq-cb480c95
    fn supports_barostat(&self, _params: &toml::Value) -> bool {
        true
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
            Box::new(RespaBuilder),
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

// rq-343a8f18 — how often a barostat couples to the dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarostatPeriodicity {
    /// Couples every step inside the captured per-step sequence
    /// (Berendsen, C-rescale).
    EveryStep,
    /// Runs a host-orchestrated move every `N` steps at a batch
    /// boundary (Monte-Carlo); `apply` is a no-op.
    EveryNSteps(u32),
}

// rq-076617ab
pub trait Barostat: std::fmt::Debug + Send {
    // rq-343a8f18 — declares per-step vs periodic coupling. Default
    // per-step; the Monte-Carlo barostat returns `EveryNSteps`.
    fn periodicity(&self) -> BarostatPeriodicity {
        BarostatPeriodicity::EveryStep
    }

    // rq-1179e42f — per-step coupling. A periodic barostat leaves this
    // at the no-op default and implements `apply_move` instead.
    fn apply(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _sim_box: &mut SimulationBox,
        _dt: Real,
        _timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        Ok(())
    }

    /// Perform a periodic barostat's host-orchestrated move at a batch
    /// boundary. Receives `&mut ForceField` (unlike `apply`) because the
    /// move re-evaluates the potential energy at a trial configuration.
    /// Default no-op; per-step barostats do not override it.
    ///
    /// Contract: the caller (the runner) guarantees that
    /// `buffers.potential_energies` and `buffers.forces_*` hold the
    /// current configuration's values on entry — i.e. the force
    /// evaluation on the step immediately preceding the move ran at
    /// `AggregateLevel::ForcesAndScalars`. A periodic barostat relies on
    /// this to obtain the current-configuration energy without a
    /// redundant force evaluation. See `rqm/integration/mc-barostat.md`.
    fn apply_move(
        &mut self,
        _force_field: &mut ForceField,
        _buffers: &mut ParticleBuffers,
        _sim_box: &mut SimulationBox,
        _constraint: Option<&mut dyn Constraint>,
        _dt: Real,
        _timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        Ok(())
    }

    /// One-time per-run initialisation invoked by the runner after
    /// construction, once the simulation box and the connectivity-derived
    /// molecule partition are available. Default no-op; the Monte-Carlo
    /// barostat uploads its molecule tables and resolves its default
    /// volume step here.
    fn init_run(
        &mut self,
        _sim_box: &SimulationBox,
        _molecules: &MoleculeList,
    ) -> Result<(), BarostatError> {
        Ok(())
    }

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
            Box::new(McBarostatBuilder),
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

