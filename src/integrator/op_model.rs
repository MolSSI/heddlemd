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
    /// this step.
    ReadsStaleResource {
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
        }
    }
}

impl std::error::Error for ScheduleError {}

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

// rq-c5c17dcd
/// Validate a schedule (as an ordered slice of `(op_name, footprint)`)
/// against the carried-in valid set. Returns the first
/// [`ScheduleError`], or `Ok(())` for a dependency-correct schedule.
///
/// For each operation, in order: every read must currently be valid;
/// then, if the operation writes `Positions` or `Box`, every derived
/// resource is invalidated; then the operation's writes become valid.
/// The read check runs against the valid set as it stands *before* the
/// operation's own invalidation and writes.
pub(crate) fn validate_footprints(
    ops: impl Iterator<Item = (&'static str, OpFootprint)>,
) -> Result<(), ScheduleError> {
    let mut valid = carried_in_valid_set();
    for (index, (op, fp)) in ops.enumerate() {
        // Read check (against the pre-operation valid set).
        for r in fp.reads.iter() {
            if !valid.contains(r) {
                return Err(ScheduleError::ReadsStaleResource { index, op, resource: r });
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
    Ok(())
}
