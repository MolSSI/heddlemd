// rq-846bdb8b
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, coulomb_pair_force,
};
use crate::kernels;
use crate::io::config::CoulombConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::NeighborListError;
use super::{
    AggregateLevel, CutoffHandling, ForceFieldContext, ForceFieldError, KernelArgType,
    KernelArg, KernelArgBinder, KernelArgSchema, PairForceBindContext,
    PairForceFragment, ForceLaunchBuilder, Potential, PotentialBuildContext,
    PotentialBuilder, SlotOutputView,
};
use crate::gpu::K_COULOMB_F32;
use crate::precision::Real;

// CoulombParameters carries the runtime real-space parameters: the
// cutoff and the inner switching radius. Per-particle charges live on
// `ParticleBuffers`. See rq-bfd7004c.
// rq-6bdfdd6d
#[derive(Debug, Clone, Copy)]
pub struct CoulombParameters {
    pub cutoff: Real,
    pub r_switch: Real,
}

impl From<&CoulombConfig> for CoulombParameters {
    fn from(c: &CoulombConfig) -> Self {
        CoulombParameters {
            cutoff: c.cutoff as Real,
            r_switch: c.r_switch as Real,
        }
    }
}

// rq-846bdb8b rq-d340b338
#[derive(Debug)]
pub struct CoulombState {
    #[allow(dead_code)]
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) params: CoulombParameters,
    pub(crate) exclusions: DeviceExclusionList,
    pub(crate) particle_count: usize,
    pub(crate) max_neighbors: u32,
}

impl CoulombState {
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        params: CoulombParameters,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        Ok(CoulombState {
            device,
            params,
            exclusions,
            particle_count,
            max_neighbors,
        })
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}

impl Potential for CoulombState {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn max_cutoff(&self) -> Option<Real> {
        Some(self.params.cutoff)
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
            .expect("CoulombState requires a shared neighbor list");
        timings.kernel_start(KernelStage::COULOMB_PAIR_FORCE)?;
        coulomb_pair_force(
            buffers,
            &mut output,
            sim_box,
            self.params.cutoff,
            self.params.r_switch,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_coul_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
            self.max_neighbors,
            level,
        )?;
        timings.kernel_stop(KernelStage::COULOMB_PAIR_FORCE)?;
        Ok(())
    }

    fn bind_pair_force_args(
        &self,
        _ctx: &PairForceBindContext<'_>,
        builder: &mut ForceLaunchBuilder,
    ) {
        // Validated against `coulomb_arg_schema()` — the same schema that
        // generates the fragment's entry-point args and functor-init
        // source — so the binding cannot drift from the kernel signature.
        let schema = coulomb_arg_schema();
        let mut b = KernelArgBinder::new(&schema, LABEL, builder);
        b.scalar_real("coul_k_coulomb", K_COULOMB_F32);
        b.scalar_real("coul_cutoff", self.params.cutoff);
        b.scalar_real("coul_r_switch", self.params.r_switch);
        b.buffer("coul_excl_offsets", &self.exclusions.atom_excl_offsets);
        b.buffer("coul_excl_partners", &self.exclusions.atom_excl_partners);
        b.buffer("coul_excl_scales", &self.exclusions.atom_excl_coul_scales);
        b.finish();
    }
}

/// The slot's stable label, shared by `Potential::label`, the fragment,
/// and the argument schema.
const LABEL: &str = "coulomb";

/// Single source of truth for the truncated-Coulomb pair-force kernel
/// arguments. The fragment's `entry_point_args` and `functor_init_source`
/// are generated from this list, and `bind_pair_force_args` is validated
/// against it, so the three pieces cannot drift apart.
fn coulomb_arg_schema() -> KernelArgSchema {
    use KernelArgType::{ConstPtrReal, ConstPtrU32, ScalarReal};
    KernelArgSchema::pair_force(
        LABEL,
        vec![
            KernelArg::new("coul_k_coulomb", ScalarReal, "k_coulomb"),
            KernelArg::new("coul_cutoff", ScalarReal, "cutoff"),
            KernelArg::new("coul_r_switch", ScalarReal, "r_switch"),
            KernelArg::new("coul_excl_offsets", ConstPtrU32, "excl_offsets"),
            KernelArg::new("coul_excl_partners", ConstPtrU32, "excl_partners"),
            KernelArg::new("coul_excl_scales", ConstPtrReal, "excl_scales"),
        ],
    )
}

// rq-e8550f96
#[derive(Debug, Clone)]
pub struct CoulombBuilder;

impl PotentialBuilder for CoulombBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        let Some(coul) = cx.coulomb_config else {
            return Ok(None);
        };
        let params = CoulombParameters::from(coul);
        let max_neighbors = super::max_neighbors_from(cx.neighbor_list_config, cx.particle_count);
        let state = CoulombState::new(
            cx.gpu,
            cx.particle_count,
            params,
            max_neighbors,
            cx.exclusion_list,
        )?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }

    fn pair_force_fragment(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<PairForceFragment>, ForceFieldError> {
        let Some(coul_cfg) = cx.coulomb_config else {
            return Ok(None);
        };
        let cutoff = coul_cfg.cutoff as Real;
        Ok(Some(coulomb_pair_force_fragment(cutoff)))
    }
}

/// Truncated Coulomb with CHARMM C¹ switching fragment for the
/// JIT-composed pair-force kernel.
pub fn coulomb_pair_force_fragment(cutoff: Real) -> PairForceFragment {
    let functor_source = r#"
struct CoulombPairFunctor {
    Real k_coulomb;
    Real cutoff;
    Real r_switch;
    const unsigned int *excl_offsets;
    const unsigned int *excl_partners;
    const Real *excl_scales;

    __device__ inline Real cutoff_squared(unsigned int, unsigned int) const {
        return cutoff * cutoff;
    }

    __device__ inline void evaluate(
        Real r2, Real inv_r, Real r,
        Real qi, Real qj,
        unsigned int i, unsigned int j,
        Real &factor, Real &energy, Real &virial) const
    {
        Real qq = qi * qj;
        Real inv_r2 = inv_r * inv_r;
        energy = k_coulomb * qq * inv_r;
        factor = k_coulomb * qq * inv_r * inv_r2;
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
    }

    __device__ inline Real exclusion_scale(unsigned int i, unsigned int j) const {
        return heddle_jit_exclusion_scale(i, j, excl_offsets, excl_partners, excl_scales);
    }
};
"#;
    // `entry_point_args` and `functor_init_source` are generated from
    // `coulomb_arg_schema()`, the same schema `bind_pair_force_args` is
    // validated against; the functor field names in `functor_source`
    // above must match the schema's `functor_field` entries.
    let schema = coulomb_arg_schema();
    PairForceFragment {
        label: LABEL,
        functor_struct_name: "CoulombPairFunctor",
        functor_source: functor_source.to_string(),
        entry_point_args: schema.entry_point_args(),
        functor_init_source: schema.functor_init_source(),
        cutoff: CutoffHandling::Uniform(cutoff),
    }
}

// rq-2093594f rq-846bdb8b
#[derive(Debug, Clone)]
pub struct CoulombKernels {
    pub coulomb_pair_force_f: CudaFunction,
    pub coulomb_pair_force_fev: CudaFunction,
}

impl CoulombKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::COULOMB),
            "coulomb",
            &["coulomb_pair_force_f", "coulomb_pair_force_fev"],
        )?;
        Ok(CoulombKernels {
            coulomb_pair_force_f: get_func(device, "coulomb", "coulomb_pair_force_f")?,
            coulomb_pair_force_fev: get_func(device, "coulomb", "coulomb_pair_force_fev")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The exact CUDA argument declarations and functor-init assignments
    // the composer expects for the Coulomb slot. The schema-generated
    // output MUST equal these byte-for-byte so the composed JIT kernel
    // source — and therefore the bit-wise reproducible result — is
    // unchanged.
    const EXPECTED_ENTRY_POINT_ARGS: &str = r#"    Real coul_k_coulomb,
    Real coul_cutoff,
    Real coul_r_switch,
    const unsigned int *coul_excl_offsets,
    const unsigned int *coul_excl_partners,
    const Real *coul_excl_scales,
"#;

    const EXPECTED_FUNCTOR_INIT_SOURCE: &str = r#"    composite.functor_coulomb.k_coulomb = coul_k_coulomb;
    composite.functor_coulomb.cutoff = coul_cutoff;
    composite.functor_coulomb.r_switch = coul_r_switch;
    composite.functor_coulomb.excl_offsets = coul_excl_offsets;
    composite.functor_coulomb.excl_partners = coul_excl_partners;
    composite.functor_coulomb.excl_scales = coul_excl_scales;
"#;

    #[test]
    fn generated_entry_point_args_match_expected() {
        assert_eq!(coulomb_arg_schema().entry_point_args(), EXPECTED_ENTRY_POINT_ARGS);
    }

    #[test]
    fn generated_functor_init_source_matches_expected() {
        assert_eq!(
            coulomb_arg_schema().functor_init_source(),
            EXPECTED_FUNCTOR_INIT_SOURCE
        );
    }
}
