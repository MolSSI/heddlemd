// rq-a5a919df
use std::sync::Arc;

use crate::gpu::{LennardJonesParameterTable, ParticleBuffers};
use crate::pbc::SimulationBox;
use crate::timings::Timings;

use super::topology::DeviceExclusionList;
use super::{
    AggregateLevel, CutoffHandling, ForceFieldContext, ForceFieldError, ForceLaunchBuilder,
    FragmentPasses, JitParticipant, KernelArg, KernelArgBinder, KernelArgSchema, KernelArgType,
    PairForceBindContext, PairForceFragment, PairForcePotential, Potential,
    PotentialBuildContext, PotentialBuilder, PotentialConfigEntry, PotentialParamsCategory,
    PotentialParamsClaim, SlotOutputView,
};
use crate::io::config::{ConfigError, PairInteractionConfig, translate_params_error_local};
use crate::precision::Real;

/// The `kind` string the Lennard-Jones builder claims in
/// `[[pair_interactions]]`.
pub const LJ_KIND: &str = "lennard-jones";

// rq-9244aae4
/// Typed per-entry parameter struct the Lennard-Jones builder
/// deserialises from a claimed `kind = "lennard-jones"` entry's
/// `params`. `deny_unknown_fields` is what rejects an unrecognised
/// field under the entry. An omitted `r_switch` stays `None` through
/// conversion and is resolved to `0.9 * cutoff` at build time (after
/// conversion, so the dimensionless ratio multiplies the
/// already-converted cutoff).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct LjPairParams {
    pub sigma: crate::units::Length,
    pub epsilon: crate::units::Energy,
    #[serde(default)]
    pub r_switch: Option<crate::units::Length>,
}

/// A `kind = "lennard-jones"` pair entry with its typed params
/// deserialised and the `r_switch` default resolved against the
/// entry's common `cutoff`.
#[derive(Debug, Clone)]
pub struct ResolvedLjPair {
    pub between: (String, String),
    pub sigma: f64,
    pub epsilon: f64,
    pub cutoff: f64,
    pub r_switch: f64,
}

/// Resolve one `[[pair_interactions]]` entry into LJ parameters.
/// Returns `None` for entries of other kinds (they belong to other
/// claiming builders). Panics on a params table that does not
/// deserialise into `LjPairParams` — validation has already
/// guaranteed the shape, so a failure here is an internal error.
// rq-1adf5954
pub fn resolve_lj_pair(p: &PairInteractionConfig) -> Option<ResolvedLjPair> {
    if p.kind != LJ_KIND {
        return None;
    }
    let params: LjPairParams = p
        .params
        .clone()
        .try_into()
        .expect("validated lennard-jones entry params (config invariant)");
    let r_switch = params.r_switch.map(|x| x.0).unwrap_or(0.9 * p.cutoff);
    Some(ResolvedLjPair {
        between: p.between.clone(),
        sigma: params.sigma.0,
        epsilon: params.epsilon.0,
        cutoff: p.cutoff,
        r_switch,
    })
}

// rq-af2d1628
#[derive(Debug)]
pub struct LennardJonesState {
    pub(crate) params: LennardJonesParameterTable,
    /// Clone of the `ForceField`'s shared device exclusion list; the
    /// LJ functor consumes `atom_excl_lj_scales`. rq-a5a919df
    pub(crate) exclusions: Arc<DeviceExclusionList>,
    pub(crate) particle_count: usize,
    pub(crate) max_cutoff: Real,
    /// `true` when every configured pair-interaction has
    /// `r_switch == cutoff`, making the switch polynomial unreachable.
    /// Drives the fragment's evaluate body.
    pub(crate) switch_degenerate: bool,
    /// `Some(c)` when every pair-interaction shares cutoff `c`
    /// (fragment reports `CutoffHandling::Uniform(c)`); `None` for a
    /// mixed-cutoff table (`CutoffHandling::PerPair`).
    pub(crate) uniform_cutoff: Option<Real>,
}

impl LennardJonesState {
    pub fn new(
        particle_count: usize,
        params: LennardJonesParameterTable,
        max_cutoff: Real,
        exclusions: Arc<DeviceExclusionList>,
        switch_degenerate: bool,
        uniform_cutoff: Option<Real>,
    ) -> Self {
        LennardJonesState {
            params,
            exclusions,
            particle_count,
            max_cutoff,
            switch_degenerate,
            uniform_cutoff,
        }
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}

impl Potential for LennardJonesState {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn max_cutoff(&self) -> Option<Real> {
        Some(self.max_cutoff)
    }

    fn compute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        mut output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        // LennardJonesState is always a JIT pair-force participant
        // (see `jit_participant`); the framework evaluates it through the
        // composed packed pair-force kernel and skips this slot in the
        // per-slot `compute` loop, so this method is never invoked.
        let _ = (buffers, sim_box, &mut output, cx, &mut *timings, level);
        unreachable!("LennardJonesState is JIT-composed; compute() is never invoked")
    }

    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        Some(JitParticipant::PairForce(self))
    }
}

impl PairForcePotential for LennardJonesState {
    fn pair_force_fragment(&self) -> PairForceFragment {
        lj_pair_force_fragment(self.switch_degenerate, self.uniform_cutoff)
    }

    fn bind_pair_force_args(
        &self,
        _ctx: &PairForceBindContext<'_>,
        builder: &mut ForceLaunchBuilder,
    ) {
        // Each push is validated against `lj_arg_schema()` — the same
        // schema that GENERATES the fragment's entry-point args and
        // functor-init source — so the binding cannot drift in order,
        // name, kind, or element type from the kernel signature.
        let schema = lj_arg_schema();
        let mut b = KernelArgBinder::new(&schema, LABEL, builder);
        // The per-atom type index is a composer common argument loaded
        // once per atom by the outer loop; the fragment receives it as
        // i_type / j_type and binds no type-index buffer of its own.
        // rq-08f7531f rq-62a18360
        b.scalar_u32("lj_n_types", self.params.n_types as u32);
        b.buffer("lj_type_sigma", &self.params.sigma);
        b.buffer("lj_type_epsilon", &self.params.epsilon);
        b.buffer("lj_type_cutoff", &self.params.cutoff);
        b.buffer("lj_type_switch", &self.params.switch);
        b.buffer("lj_excl_offsets", &self.exclusions.atom_excl_offsets);
        b.buffer("lj_excl_partners", &self.exclusions.atom_excl_partners);
        b.buffer("lj_excl_scales", &self.exclusions.atom_excl_lj_scales);
        b.finish();
    }
}

/// The slot's stable label, shared by `Potential::label`, the fragment,
/// and the argument schema.
const LABEL: &str = "lennard_jones";

/// Single source of truth for the LJ pair-force kernel arguments. The
/// fragment's `entry_point_args` and `functor_init_source` are generated
/// from this list, and `bind_pair_force_args` is validated against it,
/// so the three pieces cannot drift apart. The order here defines the
/// kernel's parameter order; each entry pairs a CUDA parameter name and
/// type with the `LjPairFunctor` field it initialises.
fn lj_arg_schema() -> KernelArgSchema {
    use KernelArgType::{ConstPtrReal, ConstPtrU32, ScalarU32};
    KernelArgSchema::pair_force(
        LABEL,
        vec![
            KernelArg::new("lj_n_types", ScalarU32, "n_types"),
            KernelArg::new("lj_type_sigma", ConstPtrReal, "type_sigma"),
            KernelArg::new("lj_type_epsilon", ConstPtrReal, "type_epsilon"),
            KernelArg::new("lj_type_cutoff", ConstPtrReal, "type_cutoff"),
            KernelArg::new("lj_type_switch", ConstPtrReal, "type_switch"),
            KernelArg::new("lj_excl_offsets", ConstPtrU32, "excl_offsets"),
            KernelArg::new("lj_excl_partners", ConstPtrU32, "excl_partners"),
            KernelArg::new("lj_excl_scales", ConstPtrReal, "excl_scales"),
        ],
    )
}

// rq-e8550f96
#[derive(Debug, Clone)]
pub struct LennardJonesBuilder;

impl PotentialBuilder for LennardJonesBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        // rq-be18633a — the slot is present when the config carries any LJ
        // configuration: at least one `kind = "lennard-jones"` entry, a
        // [lennard_jones] combining table, or both. A config with neither
        // has no LJ interaction.
        let lj_pairs: Vec<ResolvedLjPair> = cx
            .pair_interactions
            .iter()
            .filter_map(resolve_lj_pair)
            .collect();
        if lj_pairs.is_empty() && cx.lennard_jones.is_none() {
            return Ok(None);
        }
        let params = LennardJonesParameterTable::from_config(
            &cx.gpu.device,
            cx.particle_types,
            &lj_pairs,
            cx.lennard_jones,
        )?;
        // rq-be18633a — cutoff structure spans both the per-pair entry
        // cutoffs and the [lennard_jones] combined cutoff. `cutoffs`
        // collects every cutoff/r_switch that appears in the resolved
        // table; `max_cutoff` is their maximum, `uniform_cutoff` is
        // `Some(c)` only when they all share one value, and
        // `switch_degenerate` holds only when every one has
        // `r_switch == cutoff`.
        let cutoffs: Vec<(f64, f64)> = lj_pairs
            .iter()
            .map(|p| (p.cutoff, p.r_switch))
            .chain(cx.lennard_jones.map(|lj| (lj.cutoff, lj.r_switch)))
            .collect();
        let max_cutoff = cutoffs
            .iter()
            .map(|(c, _)| *c as Real)
            .fold(0.0, Real::max);
        let first_cutoff = cutoffs[0].0;
        let cutoff_uniform = cutoffs
            .iter()
            .all(|(c, _)| (c - first_cutoff).abs() < f64::EPSILON);
        let switch_degenerate = cutoffs
            .iter()
            .all(|(c, rs)| (rs - c).abs() < f64::EPSILON);
        let uniform_cutoff = if cutoff_uniform {
            Some(first_cutoff as Real)
        } else {
            None
        };
        let state = LennardJonesState::new(
            cx.particle_count,
            params,
            max_cutoff,
            cx.device_exclusions.clone(),
            switch_degenerate,
            uniform_cutoff,
        );
        Ok(Some(Box::new(state)))
    }

    // rq-529b9e4b rq-35a89768
    fn params_claim(&self) -> Option<PotentialParamsClaim> {
        Some(PotentialParamsClaim {
            category: PotentialParamsCategory::PairInteraction,
            kind: LJ_KIND,
        })
    }

    // rq-9244aae4 rq-95a50109 — sigma/epsilon finite and >= 0 (a zero
    // is Lennard-Jones-inert, negatives/NaN/inf are rejected);
    // r_switch, when supplied, finite, strictly positive, and <= the
    // entry's common cutoff. Errors carry entry-relative paths; the
    // loader prefixes the document location.
    fn validate_params(
        &self,
        entry: PotentialConfigEntry<'_>,
    ) -> Result<(), ConfigError> {
        let PotentialConfigEntry::PairInteraction(p) = entry else {
            unreachable!("dispatch guarantees the claimed category");
        };
        let params: LjPairParams = p
            .params
            .clone()
            .try_into()
            .map_err(translate_params_error_local)?;
        require_finite_non_negative("sigma", params.sigma.0)?;
        require_finite_non_negative("epsilon", params.epsilon.0)?;
        if let Some(rs) = params.r_switch {
            let rs = rs.0;
            if !rs.is_finite() || rs <= 0.0 {
                return Err(ConfigError::InvalidValue {
                    field: "r_switch".to_string(),
                    reason: "must be finite and strictly positive".to_string(),
                });
            }
            if rs > p.cutoff {
                return Err(ConfigError::InvalidValue {
                    field: "r_switch".to_string(),
                    reason: format!("r_switch ({rs}) exceeds cutoff ({})", p.cutoff),
                });
            }
        }
        Ok(())
    }

    // rq-529b9e4b
    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), ConfigError> {
        crate::registry::convert_params_in_place::<LjPairParams>(units, params)
    }
}

fn require_finite_non_negative(field: &str, v: f64) -> Result<(), ConfigError> {
    if !v.is_finite() || v < 0.0 {
        return Err(ConfigError::InvalidValue {
            field: field.to_string(),
            reason: "must be finite and >= 0".to_string(),
        });
    }
    Ok(())
}

/// LJ-12-6 (optionally with CHARMM C¹ switching) fragment for the
/// JIT-composed pair-force kernel. The functor reads per-type
/// parameters from device buffers, computes the per-pair
/// force / energy / virial, and looks up the LJ exclusion scale from
/// the slot's own exclusion table.
///
/// `switch_degenerate = true` selects the no-switch evaluate body
/// (every configured pair-interaction has `r_switch == cutoff`,
/// making the switch polynomial unreachable). `uniform_cutoff` sets
/// the fragment's `CutoffHandling`: `Some(c)` reports
/// `Uniform(c)` when every configured pair-interaction shares the
/// same cutoff; `None` reports `PerPair`. The functor struct fields
/// and the entry-point argument list are identical in both cases —
/// only the evaluate body differs — so `bind_pair_force_args` does
/// not branch on these flags.
pub fn lj_pair_force_fragment(
    switch_degenerate: bool,
    uniform_cutoff: Option<Real>,
) -> PairForceFragment {
    let evaluate_body = if switch_degenerate {
        // No-switch path: every type-pair has r_switch == cutoff so
        // the chain-rule branch is unreachable. Emit only the
        // unmodified Lennard-Jones expression. `cutoff` and
        // `r_switch` are not read in the body even though the
        // functor still carries the pointers (so the slot's
        // bind_pair_force_args is identical to the with-switch
        // case).
        r#"        unsigned int p = slot(i_type, j_type);
        Real sigma = type_sigma[p];
        Real epsilon = type_epsilon[p];
        Real inv_r2 = inv_r * inv_r;
        Real sigma2 = sigma * sigma;
        Real sr2 = sigma2 * inv_r2;
        Real sr6 = sr2 * sr2 * sr2;
        Real sr12 = sr6 * sr6;
        factor = R(24.0) * epsilon * inv_r2 * (R(2.0) * sr12 - sr6);
        energy = R(4.0) * epsilon * (sr12 - sr6);
"#
    } else {
        r#"        unsigned int p = slot(i_type, j_type);
        Real sigma = type_sigma[p];
        Real epsilon = type_epsilon[p];
        Real cutoff = type_cutoff[p];
        Real r_switch = type_switch[p];
        Real inv_r2 = inv_r * inv_r;
        Real sigma2 = sigma * sigma;
        Real sr2 = sigma2 * inv_r2;
        Real sr6 = sr2 * sr2 * sr2;
        Real sr12 = sr6 * sr6;
        factor = R(24.0) * epsilon * inv_r2 * (R(2.0) * sr12 - sr6);
        energy = R(4.0) * epsilon * (sr12 - sr6);
        Real r_s2 = r_switch * r_switch;
        Real r_c2 = cutoff * cutoff;
        Real delta = r_c2 - r_s2;
        // Skip the switch polynomial when there is no switching
        // window (r_switch == cutoff for this type pair). With
        // delta == 0, `1 / delta` would produce inf/NaN; the outer-
        // loop cutoff_mask handles the hard discontinuity at r =
        // cutoff. The Rust-side `switch_degenerate` shortcut already
        // elides this branch when EVERY pair-type has r_switch ==
        // cutoff, but this in-kernel guard catches mixed-mode
        // configurations where only some pair-types are degenerate.
        if (delta > R(0.0) && r2 > r_s2) {
            Real inv_delta = R(1.0) / delta;
            Real tau = (r2 - r_s2) * inv_delta;
            Real one_minus_tau = R(1.0) - tau;
            Real s = one_minus_tau * one_minus_tau * (R(1.0) + R(2.0) * tau);
            Real chain_coeff = R(12.0) * tau * one_minus_tau * inv_delta;
            factor = s * factor + chain_coeff * energy;
            energy = s * energy;
        }
"#
    };
    let functor_source = format!(
        r#"
struct LjPairFunctor {{
    unsigned int n_types;
    const Real *type_sigma;
    const Real *type_epsilon;
    const Real *type_cutoff;
    const Real *type_switch;
    const unsigned int *excl_offsets;
    const unsigned int *excl_partners;
    const Real *excl_scales;

    // The per-pair-type slot is resolved from the per-atom type indices
    // the composer's outer loop supplies (i_type / j_type), so the
    // functor performs no per-pair type-index load. rq-62a18360
    __device__ inline unsigned int slot(unsigned int ti, unsigned int tj) const {{
        return ti * n_types + tj;
    }}

    __device__ inline Real cutoff_squared(
        unsigned int i_type, unsigned int j_type,
        unsigned int /*i*/, unsigned int /*j*/) const {{
        Real c = type_cutoff[slot(i_type, j_type)];
        return c * c;
    }}

    __device__ inline void evaluate(
        Real r2, Real inv_r, Real r,
        Real /*qi*/, Real /*qj*/,
        unsigned int i_type, unsigned int j_type,
        unsigned int /*i*/, unsigned int /*j*/,
        Real &factor, Real &energy) const
    {{
{eval_body}    }}

    __device__ inline Real exclusion_scale(unsigned int i, unsigned int j) const {{
        return heddle_jit_exclusion_scale(i, j, excl_offsets, excl_partners, excl_scales);
    }}
}};
"#,
        eval_body = evaluate_body,
    );
    // The entry-point argument declarations and the functor-field
    // initialisation are GENERATED from `lj_arg_schema()`, the same
    // schema `bind_pair_force_args` is validated against. The functor
    // struct field names in `functor_source` above must match the
    // schema's `functor_field` entries (the CUDA compiler catches any
    // mismatch there); the kernel parameter order and binding order are
    // now guaranteed identical by construction.
    let schema = lj_arg_schema();
    let cutoff = match uniform_cutoff {
        Some(c) => CutoffHandling::Uniform(c),
        None => CutoffHandling::PerPair,
    };
    PairForceFragment {
        label: LABEL,
        functor_struct_name: "LjPairFunctor",
        functor_source,
        entry_point_args: schema.entry_point_args(),
        functor_init_source: schema.functor_init_source(),
        cutoff,
        passes: FragmentPasses::NeighbourListAndCorrection,
        // rq-08f7531f — LJ resolves its per-type-pair tables from the
        // per-atom type indices, so the composer must load them.
        consumes_type_index: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::{KernelArgType, KernelArg, KernelArgBinder, KernelArgSchema,
        ForceLaunchBuilder};

    // The exact CUDA strings the slot's schema generates. The per-atom
    // `type_indices` is a composer common argument (loaded once per atom
    // by the outer loop and passed to the functor as i_type / j_type),
    // so it is NOT among the slot's own bound parameters here. rq-62a18360
    const LEGACY_ENTRY_POINT_ARGS: &str = r#"    unsigned int lj_n_types,
    const Real *lj_type_sigma,
    const Real *lj_type_epsilon,
    const Real *lj_type_cutoff,
    const Real *lj_type_switch,
    const unsigned int *lj_excl_offsets,
    const unsigned int *lj_excl_partners,
    const Real *lj_excl_scales,
"#;

    const LEGACY_FUNCTOR_INIT_SOURCE: &str = r#"    composite.functor_lennard_jones.n_types = lj_n_types;
    composite.functor_lennard_jones.type_sigma = lj_type_sigma;
    composite.functor_lennard_jones.type_epsilon = lj_type_epsilon;
    composite.functor_lennard_jones.type_cutoff = lj_type_cutoff;
    composite.functor_lennard_jones.type_switch = lj_type_switch;
    composite.functor_lennard_jones.excl_offsets = lj_excl_offsets;
    composite.functor_lennard_jones.excl_partners = lj_excl_partners;
    composite.functor_lennard_jones.excl_scales = lj_excl_scales;
"#;

    #[test]
    fn generated_entry_point_args_match_legacy() {
        assert_eq!(lj_arg_schema().entry_point_args(), LEGACY_ENTRY_POINT_ARGS);
    }

    #[test]
    fn generated_functor_init_source_matches_legacy() {
        assert_eq!(
            lj_arg_schema().functor_init_source(),
            LEGACY_FUNCTOR_INIT_SOURCE
        );
    }

    // rq-08f7531f
    #[test]
    fn lj_fragment_declares_consumes_type_index() {
        for switch_degenerate in [false, true] {
            for uniform in [Some(1.0 as Real), None] {
                let frag = lj_pair_force_fragment(switch_degenerate, uniform);
                assert!(
                    frag.consumes_type_index,
                    "LJ must consume the per-atom type index"
                );
            }
        }
    }

    // rq-62a18360
    #[test]
    fn lj_functor_uses_per_atom_types_and_loads_no_type_indices() {
        let frag = lj_pair_force_fragment(false, Some(1.0 as Real));
        // The parameter slot is resolved from the per-atom type indices.
        assert!(
            frag.functor_source.contains("slot(i_type, j_type)"),
            "LJ functor must resolve its slot from i_type / j_type"
        );
        // The functor performs no per-pair type_indices load of its own,
        // and binds no type-index buffer (it is a composer common arg).
        assert!(
            !frag.functor_source.contains("type_indices"),
            "LJ functor must not reference a type_indices buffer"
        );
        assert!(
            !frag.entry_point_args.contains("type_indices"),
            "LJ slot must bind no type-index buffer"
        );
        assert!(
            !lj_arg_schema()
                .entry_point_args()
                .contains("type_indices"),
            "LJ arg schema must not declare a type-index buffer"
        );
    }

    // A two-scalar schema lets us exercise the binder's validation
    // without a CUDA device (scalar pushes need no buffer).
    fn two_scalar_schema() -> KernelArgSchema {
        KernelArgSchema::pair_force(
            "test",
            vec![
                KernelArg::new("a", KernelArgType::ScalarU32, "a"),
                KernelArg::new("b", KernelArgType::ScalarU32, "b"),
            ],
        )
    }

    #[test]
    fn binder_accepts_matching_schema() {
        let schema = two_scalar_schema();
        let mut builder = ForceLaunchBuilder::new();
        let mut b = KernelArgBinder::new(&schema, "test", &mut builder);
        b.scalar_u32("a", 1);
        b.scalar_u32("b", 2);
        b.finish();
    }

    #[test]
    #[should_panic(expected = "order/name drift")]
    fn binder_rejects_name_drift() {
        let schema = two_scalar_schema();
        let mut builder = ForceLaunchBuilder::new();
        let mut b = KernelArgBinder::new(&schema, "test", &mut builder);
        // Pushing "b" first is exactly the silent argument-swap bug the
        // schema is meant to catch; now it is a located panic.
        b.scalar_u32("b", 2);
    }

    #[test]
    #[should_panic(expected = "Buffer parameter but binding pushed")]
    fn binder_rejects_kind_mismatch() {
        let schema = KernelArgSchema::pair_force(
            "test",
            vec![KernelArg::new("a", KernelArgType::ConstPtrReal, "a")],
        );
        let mut builder = ForceLaunchBuilder::new();
        let mut b = KernelArgBinder::new(&schema, "test", &mut builder);
        // Schema declares a pointer; pushing a scalar must be rejected.
        b.scalar_u32("a", 1);
    }

    #[test]
    #[should_panic(expected = "pushed 1 arguments but the schema declares 2")]
    fn binder_rejects_undercount() {
        let schema = two_scalar_schema();
        let mut builder = ForceLaunchBuilder::new();
        let mut b = KernelArgBinder::new(&schema, "test", &mut builder);
        b.scalar_u32("a", 1);
        b.finish();
    }
}

