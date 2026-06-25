// rq-a5a919df rq-d3a14184
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, LennardJonesParameterTable, ParticleBuffers, lj_pair_force,
};
use crate::kernels;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::NeighborListError;
use super::{
    AggregateLevel, CutoffHandling, ForceFieldContext, ForceFieldError, ForceLaunchBuilder,
    JitParticipant, KernelArg, KernelArgBinder, KernelArgSchema, KernelArgType,
    PairForceBindContext, PairForceFragment, PairForcePotential, Potential,
    PotentialBuildContext, PotentialBuilder, SlotOutputView,
};
use crate::precision::Real;

// rq-af2d1628
#[derive(Debug)]
pub struct LennardJonesState {
    #[allow(dead_code)]
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) params: LennardJonesParameterTable,
    pub(crate) exclusions: DeviceExclusionList,
    pub(crate) particle_count: usize,
    pub(crate) max_cutoff: Real,
    pub(crate) max_neighbors: u32,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        params: LennardJonesParameterTable,
        max_cutoff: Real,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
        switch_degenerate: bool,
        uniform_cutoff: Option<Real>,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        Ok(LennardJonesState {
            device,
            params,
            exclusions,
            particle_count,
            max_cutoff,
            max_neighbors,
            switch_degenerate,
            uniform_cutoff,
        })
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
        if self.particle_count == 0 {
            return Ok(());
        }
        let nl = cx
            .neighbor_list
            .expect("LennardJonesState requires a shared neighbor list");
        timings.kernel_start(KernelStage::LJ_PAIR_FORCE)?;
        lj_pair_force(
            buffers,
            &mut output,
            sim_box,
            &self.params,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_lj_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
            self.max_neighbors,
            level,
        )?;
        timings.kernel_stop(KernelStage::LJ_PAIR_FORCE)?;
        Ok(())
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
        ctx: &PairForceBindContext<'_>,
        builder: &mut ForceLaunchBuilder,
    ) {
        // Each push is validated against `lj_arg_schema()` — the same
        // schema that GENERATES the fragment's entry-point args and
        // functor-init source — so the binding cannot drift in order,
        // name, kind, or element type from the kernel signature.
        let schema = lj_arg_schema();
        let mut b = KernelArgBinder::new(&schema, LABEL, builder);
        b.buffer("lj_type_indices", &ctx.buffers.type_indices);
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
            KernelArg::new("lj_type_indices", ConstPtrU32, "type_indices"),
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
        if cx.pair_interactions.is_empty() {
            return Ok(None);
        }
        let params = LennardJonesParameterTable::from_config(
            &cx.gpu.device,
            cx.particle_types,
            cx.pair_interactions,
        )?;
        let max_cutoff = cx
            .pair_interactions
            .iter()
            .map(|p| p.cutoff as Real)
            .fold(0.0, Real::max);
        let max_neighbors = super::max_neighbors_from(cx.neighbor_list_config, cx.particle_count);
        // Inspect the configured pair interactions to decide the
        // fragment's cutoff structure and whether the LJ switching
        // function is degenerate (every entry has r_switch == cutoff).
        let first = &cx.pair_interactions[0];
        let cutoff_uniform = cx
            .pair_interactions
            .iter()
            .all(|p| (p.cutoff - first.cutoff).abs() < f64::EPSILON);
        let switch_degenerate = cx
            .pair_interactions
            .iter()
            .all(|p| (p.r_switch - p.cutoff).abs() < f64::EPSILON);
        let uniform_cutoff = if cutoff_uniform {
            Some(first.cutoff as Real)
        } else {
            None
        };
        let state = LennardJonesState::new(
            cx.gpu,
            cx.particle_count,
            params,
            max_cutoff,
            max_neighbors,
            cx.exclusion_list,
            switch_degenerate,
            uniform_cutoff,
        )?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }
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
        r#"        unsigned int p = slot(i, j);
        Real sigma = type_sigma[p];
        Real epsilon = type_epsilon[p];
        Real inv_r2 = inv_r * inv_r;
        Real sigma2 = sigma * sigma;
        Real sr2 = sigma2 * inv_r2;
        Real sr6 = sr2 * sr2 * sr2;
        Real sr12 = sr6 * sr6;
        factor = R(24.0) * epsilon * inv_r2 * (R(2.0) * sr12 - sr6);
        energy = R(4.0) * epsilon * (sr12 - sr6);
        virial = factor * r2;
"#
    } else {
        r#"        unsigned int p = slot(i, j);
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
        if (r2 > r_s2) {
            Real r_c2 = cutoff * cutoff;
            Real delta = r_c2 - r_s2;
            Real inv_delta = R(1.0) / delta;
            Real tau = (r2 - r_s2) * inv_delta;
            Real one_minus_tau = R(1.0) - tau;
            Real s = one_minus_tau * one_minus_tau * (R(1.0) + R(2.0) * tau);
            Real chain_coeff = R(12.0) * tau * one_minus_tau * inv_delta;
            factor = s * factor + chain_coeff * energy;
            energy = s * energy;
        }
        virial = factor * r2;
"#
    };
    let functor_source = format!(
        r#"
struct LjPairFunctor {{
    const unsigned int *type_indices;
    unsigned int n_types;
    const Real *type_sigma;
    const Real *type_epsilon;
    const Real *type_cutoff;
    const Real *type_switch;
    const unsigned int *excl_offsets;
    const unsigned int *excl_partners;
    const Real *excl_scales;

    __device__ inline unsigned int slot(unsigned int i, unsigned int j) const {{
        unsigned int ti = type_indices[i];
        unsigned int tj = type_indices[j];
        return ti * n_types + tj;
    }}

    __device__ inline Real cutoff_squared(unsigned int i, unsigned int j) const {{
        Real c = type_cutoff[slot(i, j)];
        return c * c;
    }}

    __device__ inline void evaluate(
        Real r2, Real inv_r, Real r,
        Real /*qi*/, Real /*qj*/,
        unsigned int i, unsigned int j,
        Real &factor, Real &energy, Real &virial) const
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
    }
}

// rq-2093594f rq-78d9fd1c
#[derive(Debug, Clone)]
pub struct LjKernels {
    pub pair_force_f: CudaFunction,
    pub pair_force_fev: CudaFunction,
}

impl LjKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::PAIR_FORCE),
            "pair_force",
            &["lj_pair_force_f", "lj_pair_force_fev"],
        )?;
        Ok(LjKernels {
            pair_force_f: get_func(device, "pair_force", "lj_pair_force_f")?,
            pair_force_fev: get_func(device, "pair_force", "lj_pair_force_fev")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::{KernelArgType, KernelArg, KernelArgBinder, KernelArgSchema,
        ForceLaunchBuilder};

    // The exact CUDA strings the slot hand-maintained before the schema
    // refactor. The generated output MUST equal these byte-for-byte so
    // the composed JIT kernel source — and therefore the bit-wise
    // reproducible result — is unchanged.
    const LEGACY_ENTRY_POINT_ARGS: &str = r#"    const unsigned int *lj_type_indices,
    unsigned int lj_n_types,
    const Real *lj_type_sigma,
    const Real *lj_type_epsilon,
    const Real *lj_type_cutoff,
    const Real *lj_type_switch,
    const unsigned int *lj_excl_offsets,
    const unsigned int *lj_excl_partners,
    const Real *lj_excl_scales,
"#;

    const LEGACY_FUNCTOR_INIT_SOURCE: &str = r#"    composite.functor_lennard_jones.type_indices = lj_type_indices;
    composite.functor_lennard_jones.n_types = lj_n_types;
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

