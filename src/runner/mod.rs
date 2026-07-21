// rq-357909e4 rq-02edd314 rq-77c1d5d9
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::forces::{
    AngleList, BondList, ConstraintList, DihedralList, ExclusionList, ForceField,
    ForceFieldError, TopologyFileError,
};
use crate::gpu::ParticleBuffers;
use crate::integrator::{
    BarostatError, ConstraintError, IntegratorError, ThermostatError,
};
use crate::io::MinlogWriterError;
use crate::minimizer::MinimizerError;
use crate::io::{
    ConfigError, InitStateError, LogWriterError, TrajectoryWriterError,
};
use crate::state::ParticleStateError;
use crate::timings::{
    TimingsError, TimingsWriterError,
};
use crate::precision::Real;

mod cli;
mod lint;
mod output;
mod phases;
mod setup;
mod step;

// Public API surface (unchanged): re-exported so external callers keep using
// `crate::runner::{...}` exactly as before the split.
pub use cli::{cli_main, cli_main_u8};
pub use lint::{
    lint_simulation, lint_simulation_with_registries, LintOverall, LintReport, LintStage,
    LintStatus,
};
pub use phases::{run_md_phase, run_minimization_phase};
pub use setup::{run_simulation, run_simulation_with_registries};

// Runner-internal cross-module surface. Re-exported here so each submodule's
// `use super::*` reaches sibling helpers through one documented list.
pub(crate) use output::{capture_physics_sample, collect_log_extras, handle_step_output, write_traj_frame};
pub(crate) use phases::{run_md_phase_inner, run_minimization_phase_inner};
pub(crate) use setup::{run_simulation_with_phase, simulation_setup_finish_gpu};
pub(crate) use step::{
    barostat_couples_per_step, capture_phase_graph, phase_slots_graph_compatible,
    run_batched_graph_loop, run_per_step_range,
};

// rq-8ee27e27 rq-e1ceb5c0 rq-6cf916af
//
// `RunnerError` carries no `From` impls: the runner attaches an
// `ExitPhase` tag at each `.map_err` call site. The wrapping variants
// delegate `Display` to the inner error via `#[error("{0}")]` and expose
// it through `source()` via `#[source]`, but deliberately omit `#[from]`,
// so no implicit conversion exists.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    #[error("{0}")]
    Config(#[source] ConfigError),
    #[error("{0}")]
    InitState(#[source] InitStateError),
    #[error("{0}")]
    ParticleState(#[source] ParticleStateError),
    #[error("{0}")]
    Gpu(#[source] crate::gpu::GpuError),
    #[error("{0}")]
    Integrator(#[source] IntegratorError),
    #[error("{0}")]
    Thermostat(#[source] ThermostatError),
    #[error("{0}")]
    Barostat(#[source] BarostatError),
    #[error("{0}")]
    SimulationBox(#[source] crate::pbc::SimulationBoxError),
    #[error("{0}")]
    Constraint(#[source] ConstraintError),
    #[error("{0}")]
    TopologyFile(#[source] TopologyFileError),
    // rq-8ee27e27 — a topology [charges] section coexists with a nonzero
    // per-type charge. The two charge sources are mutually exclusive.
    #[error(
        "particle type `{type_name}` declares a nonzero charge ({charge}) while the topology \
         file supplies a [charges] section; drop the per-type charge when using [charges]"
    )]
    TypeChargeWithPerAtomCharges { type_name: String, charge: f64 },
    #[error("{0}")]
    ForceField(#[source] ForceFieldError),
    #[error("{0}")]
    Trajectory(#[source] TrajectoryWriterError),
    #[error("{0}")]
    Log(#[source] LogWriterError),
    #[error("{0}")]
    Timings(#[source] TimingsError),
    #[error("{0}")]
    TimingsWriter(#[source] TimingsWriterError),
    #[error("missing command-line arguments")]
    MissingArgs,
    #[error("output file already exists: `{}`", .path.display())]
    OutputExists { path: PathBuf },
    #[error("simulation box perpendicular width along lattice direction `{direction}` is {width}, below the required {required}")]
    CellListBoxTooSmall {
        direction: &'static str,
        width: Real,
        required: Real,
    },
    // rq-8ee27e27 rq-02f4d342
    #[error(
        "cuFFT returned non-deterministic R2C output between two identical runs ({differences} differing floats); SPME requires bit-exact reciprocal-space behaviour"
    )]
    CuFftNonDeterministic { differences: usize },
    // rq-fd8bb824 — wraps an analyze-pipeline error for surfacing in
    // the lint / CLI paths.
    #[error("{0}")]
    Analyze(#[source] crate::analysis::AnalyzeError),
    #[error("{0}")]
    Minimizer(#[source] MinimizerError),
    #[error("{0}")]
    Minlog(#[source] MinlogWriterError),
    #[error(
        "minimization phase `{phase}` failed to converge after {iterations} iterations (max_force = {final_force:.3e} N, step = {final_step:.3e} m)"
    )]
    MinimizerNonConvergence {
        phase: String,
        iterations: u64,
        final_force: f64,
        final_step: f64,
    },
    /// A per-step barostat is configured with an integrator whose plan
    /// carries no `BarostatPoint`, so the barostat's `apply` would never
    /// fire. rq-dbbffa7d
    #[error(
        "a per-step barostat is configured but integrator `{integrator}` emits no \
         BarostatPoint in its plan; the barostat's coupling would never fire"
    )]
    BarostatPlacementMissing { integrator: String },
    /// An integrator's `StepPlan` fails schedule dependency validation
    /// (an operation reads force-derived state a preceding position/box
    /// mutation invalidated, with no intervening force evaluation). See
    /// `rqm/integration/op-model.md`. rq-77f1e6ef
    #[error("integrator `{integrator}` emits a dependency-invalid schedule: {source}")]
    InvalidSchedule {
        integrator: String,
        #[source]
        source: crate::integrator::ScheduleError,
    },
}

// rq-5c1cfc93 rq-b00170c6
#[derive(Debug, Clone)]
pub struct PhaseSummary {
    pub name: String,
    pub n_steps: u64,
    pub frames_written: u64,
    pub log_rows_written: u64,
    pub elapsed_micros: u128,
    /// Phase kind: "md" for `[[phase]]`, "minimization" for
    /// `[[minimization]]`. Used by the CLI summary formatter.
    pub kind: &'static str,
    /// For minimization phases: the convergence reason as a short
    /// token (`"force_tolerance"`, `"energy_tolerance"`,
    /// `"force_zero"`, `"step_floor"`, `"max_iterations"`). `None`
    /// for MD phases.
    pub convergence: Option<&'static str>,
    /// For minimization phases: the `max_force` at the final accepted
    /// state (N). Populated for every minimization phase; consumed by
    /// the CLI summary formatter to name the residual gradient when
    /// the convergence reason is `step_floor` (that reason does not
    /// imply `F_max ≤ force_tolerance`, so surfacing the value lets
    /// the user judge whether the state is acceptable for downstream
    /// dynamics). `None` for MD phases.
    pub min_final_max_force: Option<f64>,
    /// The phase's physics series, one entry per emitted CSV log row and
    /// in the same order; empty when `log_every == 0`. Captured from the
    /// same forces-and-scalars evaluation that produces each log row, so
    /// it adds no force evaluations beyond those logging already
    /// performs. Empty for minimization phases.
    pub physics: Vec<PhysicsSample>,
}

// rq-0286c77d — one physics snapshot of the system, carried in
// `PhaseSummary.physics`. All values are f64 in Hartree atomic units,
// computed with f64 arithmetic on f32-downloaded state (matching the
// KE/temperature convention used throughout the runner).
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsSample {
    /// Absolute step index within the phase.
    pub step: u64,
    /// Simulated time at that step (`step * dt`).
    pub time: f64,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    /// `kinetic_energy + potential_energy`.
    pub total_energy: f64,
    /// Instantaneous kinetic temperature, using the same thermal-DOF
    /// convention as `compute_temperature`.
    pub temperature: f64,
    /// Instantaneous scalar pressure `(2*KE + virial) / (3*V)`.
    pub pressure: f64,
    /// Simulation-box volume at that step.
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub phases: Vec<PhaseSummary>,
    pub total_n_steps: u64,
    pub total_elapsed_micros: u128,
}

// rq-b1a2d006 — host-stage durations captured during one-time setup.
// Replayed into phase-0 `Timings` as static one-shot samples by
// `run_md_phase` / `run_minimization_phase`.
#[derive(Debug, Clone, Default)]
pub struct PrePhaseDurations {
    pub config_load: Duration,
    pub init_load: Duration,
    pub gpu_init: Duration,
    pub velocity_generation: Duration,
    pub upload: Duration,
}

// rq-b1a2d006 — cross-phase state owned for the duration of a run.
//
// Constructed once via `SimulationSetup::new`, then mutated by the
// per-phase functions `run_md_phase` and `run_minimization_phase`.
// Every field is `pub` so external scenario-driving binaries can
// inspect or replace pieces between calls.
#[derive(Debug)]
pub struct SimulationSetup {
    pub config: crate::io::Config,
    pub registries: crate::Registries,
    pub gpu: crate::gpu::GpuContext,
    pub buffers: ParticleBuffers,
    pub sim_box: crate::pbc::SimulationBox,
    pub force_field: ForceField,
    pub constraint_list: ConstraintList,
    pub bond_list: BondList,
    pub angle_list: AngleList,
    pub dihedral_list: DihedralList,
    pub exclusion_list: ExclusionList,
    pub masses: Vec<Real>,
    pub charges: Vec<Real>,
    pub type_indices: Vec<u32>,
    pub n_constraints: u32,
    /// `max(0, 3N − n_constraints − 3)`: the constraint- and
    /// COM-removed thermal DOF count used by `compute_temperature`
    /// and the initial-velocity equipartition rescale. Zero is a
    /// legitimate value (reported temperature is then 0.0); the
    /// thermostats compute their own per-slot counts with the clamp
    /// each algorithm needs.
    pub n_thermal_dof: u32,
    pub pre_phase_durations: PrePhaseDurations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExitPhase {
    Setup,
    Loop,
}

fn compute_cutoff_max(config: &crate::io::Config) -> f64 {
    let mut cutoff_max: f64 = config
        .pair_interactions
        .iter()
        .map(|p| p.cutoff)
        .fold(0.0, f64::max);
    // rq-be18633a — combined pairs use the [lennard_jones] cutoff.
    if let Some(lj) = config.lennard_jones.as_ref() {
        cutoff_max = cutoff_max.max(lj.cutoff);
    }
    if let Some(s) = config.spme.as_ref() {
        cutoff_max = cutoff_max.max(s.r_cut_real);
    }
    cutoff_max
}

// rq-65a63eec — box dimensions and perpendicular widths are stored in
// atomic units (Bohr) internally; user-facing lint and error messages
// report them in the config's unit system. Returns the atomic→user length
// scale factor and the matching unit symbol.
fn length_display(units: crate::units::UnitSystem) -> (f64, &'static str) {
    let factor = units.to_user(crate::units::Dimension::Length, 1.0);
    let symbol = match units {
        crate::units::UnitSystem::Si => "m",
        crate::units::UnitSystem::Atomic => "a_0",
    };
    (factor, symbol)
}

fn timed<T>(target: &mut Duration, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let value = f();
    *target = started.elapsed();
    value
}
