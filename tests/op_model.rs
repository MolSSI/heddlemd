//! Op schedule dependency-model tests. Implements the Gherkin scenarios
//! in `rqm/integration/op-model.md`. `StepPlan::validate` is pure (no
//! device work), so these tests build plans directly and validate them.

use heddle_md::forces::{AggregateLevel, ForceClass};
use heddle_md::integrator::{
    ConstraintPhase, KickSource, OpFootprint, Resource, ResourceSet, ScheduleError, StepPlan,
    StepValidationContext, SubStep, ThermostatPhase,
};
use heddle_md::precision::Real;

const DT: Real = 0.1;

fn plan(steps: Vec<SubStep>) -> StepPlan {
    StepPlan { steps }
}

/// No-barostat context: BarostatPoint markers are inert (NVE / NVT).
fn nb() -> StepValidationContext {
    StepValidationContext::no_barostat()
}

/// Active per-step barostat context; `tolerates` is its
/// `tolerates_stale_cached_forces()` value.
fn per_step(tolerates: bool) -> StepValidationContext {
    StepValidationContext::per_step_barostat(tolerates)
}

fn kick_total() -> SubStep {
    SubStep::KickHalf { dt: DT, label: "k", source: KickSource::Total }
}
fn kickdrift_total() -> SubStep {
    SubStep::KickDrift { dt: DT, label: "kd", source: KickSource::Total }
}
fn kick_class(c: ForceClass) -> SubStep {
    SubStep::KickHalf { dt: DT, label: "kc", source: KickSource::Class(c) }
}
fn kickdrift_class(c: ForceClass) -> SubStep {
    SubStep::KickDrift { dt: DT, label: "kdc", source: KickSource::Class(c) }
}
fn drift() -> SubStep {
    SubStep::Drift { dt: DT, label: "d" }
}
fn force_eval_all() -> SubStep {
    SubStep::ForceEval { class: None, level: Some(AggregateLevel::ForcesOnly) }
}
fn force_eval_class(c: ForceClass) -> SubStep {
    SubStep::ForceEval { class: Some(c), level: None }
}
fn thermostat_pre() -> SubStep {
    SubStep::ThermostatHalf { dt: DT, phase: ThermostatPhase::Pre }
}
fn constraint(phase: ConstraintPhase) -> SubStep {
    SubStep::ConstraintPoint { phase, dt: DT }
}
fn barostat() -> SubStep {
    SubStep::BarostatPoint { dt: DT }
}
fn custom(reads: &[Resource], writes: &[Resource]) -> SubStep {
    SubStep::Custom {
        dt: DT,
        label: "custom",
        reads: ResourceSet::from_slice(reads),
        writes: ResourceSet::from_slice(writes),
    }
}

/// Assert a plan fails the intra-step pass with ReadsStaleResource for the
/// given index / op / resource (no-barostat context).
fn assert_stale(plan: &StepPlan, want_index: usize, want_op: &str, want_resource: Resource) {
    match plan.validate(&nb()) {
        Err(ScheduleError::ReadsStaleResource { index, op, resource }) => {
            assert_eq!(index, want_index, "op index");
            assert_eq!(op, want_op, "op name");
            assert_eq!(resource, want_resource, "resource");
        }
        other => panic!("expected ReadsStaleResource, got {other:?}"),
    }
}

/// Assert a plan fails the cross-step pass with ReadsStaleCachedForce for
/// the given index / op / resource under `ctx`.
fn assert_stale_cross_step(
    plan: &StepPlan,
    ctx: &StepValidationContext,
    want_index: usize,
    want_op: &str,
    want_resource: Resource,
) {
    match plan.validate(ctx) {
        Err(ScheduleError::ReadsStaleCachedForce { index, op, resource }) => {
            assert_eq!(index, want_index, "op index");
            assert_eq!(op, want_op, "op name");
            assert_eq!(resource, want_resource, "resource");
        }
        other => panic!("expected ReadsStaleCachedForce, got {other:?}"),
    }
}

#[test] // rq-1c8baf7d
fn velocity_verlet_plan_validates() {
    let p = plan(vec![
        constraint(ConstraintPhase::BeforeDrift),
        kickdrift_total(),
        constraint(ConstraintPhase::AfterDrift),
        force_eval_all(),
        kick_total(),
        constraint(ConstraintPhase::AfterKick),
        barostat(),
    ]);
    assert_eq!(p.validate(&nb()), Ok(()));
}

#[test] // rq-ab2607c7
fn empty_plan_validates() {
    assert_eq!(plan(vec![]).validate(&nb()), Ok(()));
}

#[test] // rq-0625c5d4
fn base_resources_readable_at_step_start() {
    // ThermostatHalf reads Velocities at index 0 without error.
    let p = plan(vec![thermostat_pre(), force_eval_all(), kick_total()]);
    assert_eq!(p.validate(&nb()), Ok(()));
}

#[test] // rq-7fde3409
fn cached_forces_readable_at_step_start() {
    // The leading KickDrift reads the carried-in Forces without error.
    let p = plan(vec![kickdrift_total(), force_eval_all(), kick_total()]);
    assert_eq!(p.validate(&nb()), Ok(()));
}

#[test] // rq-df53eb91
fn force_read_after_drift_with_no_eval_is_stale() {
    let p = plan(vec![kickdrift_total(), kick_total()]);
    assert_stale(&p, 1, "KickHalf", Resource::Forces);
}

#[test] // rq-b577c9bd
fn force_eval_revalidates_forces_after_position_update() {
    let p = plan(vec![kickdrift_total(), force_eval_all(), kick_total()]);
    assert_eq!(p.validate(&nb()), Ok(()));
}

#[test] // rq-a53f18a6
fn bare_drift_then_total_kick_is_stale() {
    let p = plan(vec![drift(), kick_total()]);
    assert_stale(&p, 1, "KickHalf", Resource::Forces);
}

#[test] // rq-41dab871
fn respa_plan_validates() {
    // [KickHalf{Slow}, (KickDrift{Fast}, ForceEval{Fast}, KickHalf{Fast}) x 3,
    //  ForceEval{Slow}, KickHalf{Slow}]
    let mut steps = vec![kick_class(ForceClass::Slow)];
    for _ in 0..3 {
        steps.push(kickdrift_class(ForceClass::Fast));
        steps.push(force_eval_class(ForceClass::Fast));
        steps.push(kick_class(ForceClass::Fast));
    }
    steps.push(force_eval_class(ForceClass::Slow));
    steps.push(kick_class(ForceClass::Slow));
    assert_eq!(plan(steps).validate(&nb()), Ok(()));
}

#[test] // rq-95031d97
fn class_kick_on_drift_invalidated_accumulator_is_stale() {
    let p = plan(vec![kickdrift_class(ForceClass::Fast), kick_class(ForceClass::Fast)]);
    assert_stale(&p, 1, "KickHalf", Resource::ClassForces(ForceClass::Fast));
}

#[test] // rq-12a0b0f8
fn class_specific_eval_revalidates_only_its_own_class() {
    // Fast drift invalidates both classes; a Fast eval revalidates only
    // Fast, so a Slow kick reads a stale Slow accumulator.
    let p = plan(vec![
        kickdrift_class(ForceClass::Fast),
        force_eval_class(ForceClass::Fast),
        kick_class(ForceClass::Slow),
    ]);
    assert_stale(&p, 2, "KickHalf", Resource::ClassForces(ForceClass::Slow));
}

#[test] // rq-ce497f66
fn kickdrift_reads_valid_forces_before_invalidating_them() {
    let p = plan(vec![force_eval_all(), kickdrift_total(), force_eval_all(), kick_total()]);
    assert_eq!(p.validate(&nb()), Ok(()));
}

#[test] // rq-cf364916
fn custom_reading_forces_after_drift_is_stale() {
    let p = plan(vec![
        kickdrift_total(),
        custom(&[Resource::Velocities, Resource::Forces], &[Resource::Velocities]),
    ]);
    assert_stale(&p, 1, "Custom", Resource::Forces);
}

#[test] // rq-4ab1c94e
fn custom_reading_only_base_resources_always_validates() {
    let p = plan(vec![
        kickdrift_total(),
        custom(&[Resource::Velocities], &[Resource::Velocities]),
        force_eval_all(),
        kick_total(),
    ]);
    assert_eq!(p.validate(&nb()), Ok(()));
}

#[test] // rq-1fecec44
fn footprint_reports_declared_reads_and_writes() {
    let fp = kick_class(ForceClass::Slow).footprint();
    assert_eq!(
        fp.reads,
        ResourceSet::from_slice(&[Resource::Velocities, Resource::ClassForces(ForceClass::Slow)])
    );
    assert_eq!(fp.writes, ResourceSet::from_slice(&[Resource::Velocities]));
}

/// The canonical velocity-Verlet plan with a terminal `BarostatPoint`.
fn vv_with_barostat() -> StepPlan {
    plan(vec![
        constraint(ConstraintPhase::BeforeDrift),
        kickdrift_total(),
        constraint(ConstraintPhase::AfterDrift),
        force_eval_all(),
        kick_total(),
        constraint(ConstraintPhase::AfterKick),
        barostat(),
    ])
}

/// A RESPA plan with a terminal `BarostatPoint`.
fn respa_with_barostat() -> StepPlan {
    let mut steps = vec![kick_class(ForceClass::Slow)];
    for _ in 0..3 {
        steps.push(kickdrift_class(ForceClass::Fast));
        steps.push(force_eval_class(ForceClass::Fast));
        steps.push(kick_class(ForceClass::Fast));
    }
    steps.push(force_eval_class(ForceClass::Slow));
    steps.push(kick_class(ForceClass::Slow));
    steps.push(barostat());
    plan(steps)
}

#[test] // rq-97093c72
fn untolerated_terminal_box_mutation_leaves_next_step_forces_stale() {
    // Under an active, non-tolerant per-step barostat the terminal
    // BarostatPoint invalidates the forces the next step's leading
    // KickDrift reads: the cross-step pass flags it.
    let p = vv_with_barostat();
    assert_eq!(p.validate(&nb()), Ok(())); // inert marker: fine without a barostat
    assert_stale_cross_step(&p, &per_step(false), 1, "KickDrift", Resource::Forces);
}

#[test] // rq-060b1323
fn tolerated_terminal_box_mutation_validates() {
    let p = vv_with_barostat();
    assert_eq!(p.validate(&per_step(true)), Ok(()));
}

#[test] // rq-df64e69a
fn trailing_force_eval_makes_terminal_box_mutation_loop_consistent() {
    // A ForceEval after the BarostatPoint leaves the forces valid at the
    // plan's end, so no tolerance is needed even for a non-tolerant barostat.
    let p = plan(vec![
        kickdrift_total(),
        force_eval_all(),
        kick_total(),
        barostat(),
        force_eval_all(),
    ]);
    assert_eq!(p.validate(&per_step(false)), Ok(()));
}

#[test] // rq-8e1ce2f8
fn plan_beginning_with_force_eval_is_robust_to_terminal_box_mutation() {
    // The leading ForceEval re-produces the forces before any consumer, so
    // the plan never relies on carried-in forces across the boundary.
    let p = plan(vec![
        force_eval_all(),
        kick_total(),
        kickdrift_total(),
        force_eval_all(),
        kick_total(),
        barostat(),
    ]);
    assert_eq!(p.validate(&per_step(false)), Ok(()));
}

#[test] // rq-229f0723
fn respa_untolerated_terminal_box_mutation_leaves_slow_accumulator_stale() {
    // The leading KickHalf{Slow} at index 0 reads the slow accumulator the
    // terminal barostat invalidated.
    let p = respa_with_barostat();
    assert_stale_cross_step(
        &p,
        &per_step(false),
        0,
        "KickHalf",
        Resource::ClassForces(ForceClass::Slow),
    );
}

#[test] // rq-009c28e2
fn respa_with_tolerant_barostat_validates() {
    let p = respa_with_barostat();
    assert_eq!(p.validate(&per_step(true)), Ok(()));
}

#[test] // rq-450484bb
fn effective_footprint_of_inert_barostat_point_is_empty() {
    let fp = barostat().effective_footprint(&nb());
    assert_eq!(fp, OpFootprint::new(&[], &[]));
}

#[test] // rq-13cb1367
fn effective_footprint_of_active_barostat_point_is_full() {
    let fp = barostat().effective_footprint(&per_step(false));
    assert_eq!(
        fp,
        OpFootprint::new(
            &[Resource::Velocities, Resource::Box],
            &[Resource::Positions, Resource::Velocities, Resource::Box],
        )
    );
}
