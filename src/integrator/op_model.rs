//! The Op schedule dependency model: per-operation resource footprints
//! and the dependency-validation pass over a `StepPlan`. See
//! `rqm/integration/op-model.md`.

use crate::forces::ForceClass;

// rq-f44776cb
/// One distinct component of per-particle or global simulation state that
/// the schedule tracks for dependency reasoning.
///
/// `Positions`, `Velocities`, `Images`, and `Box` are **base** resources
/// (they hold state directly and are always readable). `Forces` and
/// `ClassForces(_)` are **derived** resources: a function of the current
/// positions and box, produced by a force evaluation and made stale by a
/// write to `Positions` or `Box`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resource {
    Positions,
    Velocities,
    Images,
    Box,
    Forces,
    ClassForces(ForceClass),
}

impl Resource {
    /// Bit index of this resource in a `ResourceSet`. The seven distinct
    /// resources (four base + `Forces` + one accumulator per `ForceClass`)
    /// fit in a `u8` bitset.
    fn bit(self) -> u8 {
        match self {
            Resource::Positions => 0,
            Resource::Velocities => 1,
            Resource::Images => 2,
            Resource::Box => 3,
            Resource::Forces => 4,
            Resource::ClassForces(ForceClass::Fast) => 5,
            Resource::ClassForces(ForceClass::Slow) => 6,
        }
    }

    /// `true` iff a write to this resource makes the derived force
    /// resources stale — i.e. it is `Positions` or `Box`.
    fn invalidates_forces(self) -> bool {
        matches!(self, Resource::Positions | Resource::Box)
    }

    /// `true` iff this resource is force-derived (produced by a force
    /// evaluation, invalidated by a configuration change).
    fn is_derived(self) -> bool {
        matches!(self, Resource::Forces | Resource::ClassForces(_))
    }

    /// Every resource, in bit order. Used to seed and sweep the
    /// validator's valid set.
    const ALL: [Resource; 7] = [
        Resource::Positions,
        Resource::Velocities,
        Resource::Images,
        Resource::Box,
        Resource::Forces,
        Resource::ClassForces(ForceClass::Fast),
        Resource::ClassForces(ForceClass::Slow),
    ];
}

// rq-5cf3694c
/// A set of [`Resource`] values — the reads or writes of one operation.
/// A `u8` bitset (seven distinct resources).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceSet(u8);

impl ResourceSet {
    /// The empty set.
    pub const fn empty() -> Self {
        ResourceSet(0)
    }

    /// Build a set from a slice of resources.
    pub fn from_slice(resources: &[Resource]) -> Self {
        let mut set = ResourceSet::empty();
        for &r in resources {
            set.insert(r);
        }
        set
    }

    /// Add a resource to the set.
    pub fn insert(&mut self, r: Resource) {
        self.0 |= 1 << r.bit();
    }

    /// Remove a resource from the set.
    pub fn remove(&mut self, r: Resource) {
        self.0 &= !(1 << r.bit());
    }

    /// `true` iff the set contains `r`.
    pub fn contains(&self, r: Resource) -> bool {
        self.0 & (1 << r.bit()) != 0
    }

    /// Iterate the resources present in the set (bit order).
    pub fn iter(&self) -> impl Iterator<Item = Resource> + '_ {
        Resource::ALL.into_iter().filter(move |&r| self.contains(r))
    }
}

// rq-5414f115
/// The resource footprint of one schedule operation: the sets it reads
/// and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpFootprint {
    pub reads: ResourceSet,
    pub writes: ResourceSet,
}

impl OpFootprint {
    /// Convenience constructor from resource slices.
    pub fn new(reads: &[Resource], writes: &[Resource]) -> Self {
        OpFootprint {
            reads: ResourceSet::from_slice(reads),
            writes: ResourceSet::from_slice(writes),
        }
    }
}

// rq-3fd3777d
/// Error returned by schedule dependency validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    /// The operation at `index` (named by `SubStep::variant_name()`)
    /// reads `resource`, which is not valid at that point in the
    /// schedule — stale force-derived state, or a resource never produced
    /// this step. Raised by the intra-step pass.
    ReadsStaleResource {
        index: usize,
        op: &'static str,
        resource: Resource,
    },
    /// The operation at `index` reads force-derived `resource` at the start
    /// of a step that the schedule's own terminal operations invalidated,
    /// with no intervening force evaluation and no weak-coupling tolerance.
    /// Raised by the cross-step pass.
    ReadsStaleCachedForce {
        index: usize,
        op: &'static str,
        resource: Resource,
    },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::ReadsStaleResource { index, op, resource } => write!(
                f,
                "schedule operation #{index} ({op}) reads stale resource {resource:?}: \
                 it was invalidated by an earlier position/box mutation, or never \
                 produced this step (a missing force evaluation)"
            ),
            ScheduleError::ReadsStaleCachedForce { index, op, resource } => write!(
                f,
                "schedule operation #{index} ({op}) reads force-derived resource \
                 {resource:?} at the start of a step, but the schedule's own terminal \
                 operations invalidated it (a box/position mutation with no trailing \
                 force evaluation): the replayed step would consume stale cached forces. \
                 Append a force evaluation after the mutation, or declare weak-coupling \
                 tolerance on the barostat"
            ),
        }
    }
}

impl std::error::Error for ScheduleError {}

// rq-b83f8ae6
/// The phase context schedule validation is performed against. The runner
/// assembles it from the phase's configured slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepValidationContext {
    /// Whether a per-step barostat is configured for the phase. When
    /// `false`, a `BarostatPoint` marker is inert (empty footprint).
    pub per_step_barostat_active: bool,
    /// Whether the active per-step barostat accepts the force staleness its
    /// terminal rescale leaves for the next step. Meaningful only when
    /// `per_step_barostat_active` is `true`.
    pub tolerates_stale_cached_forces: bool,
}

impl StepValidationContext {
    /// No per-step barostat (NVE / NVT, or an integrator that owns its
    /// pressure coupling): `BarostatPoint` markers are inert.
    pub fn no_barostat() -> Self {
        StepValidationContext {
            per_step_barostat_active: false,
            tolerates_stale_cached_forces: false,
        }
    }

    /// A per-step barostat is active; `tolerates` is its
    /// `Barostat::tolerates_stale_cached_forces()` value.
    pub fn per_step_barostat(tolerates: bool) -> Self {
        StepValidationContext {
            per_step_barostat_active: true,
            tolerates_stale_cached_forces: tolerates,
        }
    }
}

impl Default for StepValidationContext {
    fn default() -> Self {
        StepValidationContext::no_barostat()
    }
}

// rq-c5c17dcd
/// The carried-in valid set an integrator enters a step with: every base
/// resource plus the cached force resources. Force-derived state is
/// carried across the step boundary (the symplectic-with-cached-forces
/// contract); scalar reductions are not tracked resources.
pub(crate) fn carried_in_valid_set() -> ResourceSet {
    let mut v = ResourceSet::empty();
    for r in Resource::ALL {
        v.insert(r);
    }
    v
}

/// Walk a schedule (as an ordered slice of `(op_name, footprint)`) from a
/// given `seed` valid set, checking reads and applying invalidation and
/// writes. Returns the valid set at the end of the walk, or the first
/// stale read (built by `make_err`).
///
/// For each operation, in order: every read must currently be valid;
/// then, if the operation writes `Positions` or `Box`, every derived
/// resource is invalidated; then the operation's writes become valid. The
/// read check runs against the valid set as it stands *before* the
/// operation's own invalidation and writes.
fn walk(
    ops: &[(&'static str, OpFootprint)],
    seed: ResourceSet,
    make_err: impl Fn(usize, &'static str, Resource) -> ScheduleError,
) -> Result<ResourceSet, ScheduleError> {
    let mut valid = seed;
    for (index, (op, fp)) in ops.iter().enumerate() {
        // Read check (against the pre-operation valid set).
        for r in fp.reads.iter() {
            if !valid.contains(r) {
                return Err(make_err(index, op, r));
            }
        }
        // Invalidation: a configuration change makes cached forces stale.
        if fp.writes.iter().any(|r| r.invalidates_forces()) {
            for r in Resource::ALL {
                if r.is_derived() {
                    valid.remove(r);
                }
            }
        }
        // The operation's own writes become valid.
        for r in fp.writes.iter() {
            valid.insert(r);
        }
    }
    Ok(valid)
}

// rq-c5c17dcd rq-28c539f5 rq-338550b2 rq-fd52063f
/// Validate a schedule (as an ordered slice of effective `(op_name,
/// footprint)` pairs) as a repeating loop. Returns the first
/// [`ScheduleError`], or `Ok(())` for a dependency-correct schedule.
///
/// Two passes run the same walk with different seeds:
///
/// * **Intra-step pass** — seeded with the carried-in valid set (all forces
///   valid, as on a phase's warm-up-seeded first step). A stale read here is
///   a `ReadsStaleResource`.
/// * **Cross-step pass** — seeded with the derived state the schedule
///   actually leaves valid at its end (`d_end`, the intra-step walk's end
///   set), since every step after the first inherits exactly that. A stale
///   read here is a `ReadsStaleCachedForce`. When the active per-step
///   barostat tolerates stale cached forces (a weak-coupling terminal
///   rescale), the next step may treat the pre-rescale forces as valid, so
///   the cross-step seed is the full carried-in set instead — collapsing
///   the cross-step pass to the intra-step pass.
pub(crate) fn validate_footprints(
    ops: &[(&'static str, OpFootprint)],
    ctx: &StepValidationContext,
) -> Result<(), ScheduleError> {
    // rq-28c539f5 — intra-step pass.
    let d_end = walk(ops, carried_in_valid_set(), |index, op, resource| {
        ScheduleError::ReadsStaleResource { index, op, resource }
    })?;
    // rq-fd52063f — the carry set the successor step relies on: what this
    // schedule leaves valid, or the full carried-in set when a weak-coupling
    // barostat tolerates the staleness its terminal rescale leaves behind.
    let carry = if ctx.tolerates_stale_cached_forces {
        carried_in_valid_set()
    } else {
        d_end
    };
    // rq-338550b2 — cross-step pass.
    walk(ops, carry, |index, op, resource| {
        ScheduleError::ReadsStaleCachedForce { index, op, resource }
    })?;
    Ok(())
}
