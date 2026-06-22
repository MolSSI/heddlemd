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
    AggregateLevel, ForceFieldContext, ForceFieldError, PairForceBindContext,
    PairForceFragment, PairForceLaunchBuilder, Potential, PotentialBuildContext,
    PotentialBuilder, SlotOutputView,
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
}

impl LennardJonesState {
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        params: LennardJonesParameterTable,
        max_cutoff: Real,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
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
        })
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}

impl Potential for LennardJonesState {
    fn label(&self) -> &'static str {
        "lennard_jones"
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

    fn bind_pair_force_args(
        &self,
        ctx: &PairForceBindContext<'_>,
        builder: &mut PairForceLaunchBuilder,
    ) {
        // Order MUST match `lj_pair_force_fragment`'s entry-point args
        // below.
        builder.push_device_buffer(&ctx.buffers.type_indices);
        builder.push_scalar(self.params.n_types as u32);
        builder.push_device_buffer(&self.params.sigma);
        builder.push_device_buffer(&self.params.epsilon);
        builder.push_device_buffer(&self.params.cutoff);
        builder.push_device_buffer(&self.params.switch);
        builder.push_device_buffer(&self.exclusions.atom_excl_offsets);
        builder.push_device_buffer(&self.exclusions.atom_excl_partners);
        builder.push_device_buffer(&self.exclusions.atom_excl_lj_scales);
    }
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
        let state = LennardJonesState::new(
            cx.gpu,
            cx.particle_count,
            params,
            max_cutoff,
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
        if cx.pair_interactions.is_empty() {
            return Ok(None);
        }
        Ok(Some(lj_pair_force_fragment()))
    }
}

/// LJ-12-6 + CHARMM C¹ switching fragment for the JIT-composed
/// pair-force kernel. The functor reads per-type parameters from
/// device buffers, computes the per-pair force / energy / virial, and
/// looks up the LJ exclusion scale from the slot's own exclusion
/// table.
pub fn lj_pair_force_fragment() -> PairForceFragment {
    let functor_source = r#"
struct LjPairFunctor {
    const unsigned int *type_indices;
    unsigned int n_types;
    const Real *type_sigma;
    const Real *type_epsilon;
    const Real *type_cutoff;
    const Real *type_switch;
    const unsigned int *excl_offsets;
    const unsigned int *excl_partners;
    const Real *excl_scales;

    __device__ inline unsigned int slot(unsigned int i, unsigned int j) const {
        unsigned int ti = type_indices[i];
        unsigned int tj = type_indices[j];
        return ti * n_types + tj;
    }

    __device__ inline Real cutoff_squared(unsigned int i, unsigned int j) const {
        Real c = type_cutoff[slot(i, j)];
        return c * c;
    }

    __device__ inline void evaluate(
        Real r2, unsigned int i, unsigned int j,
        Real &factor, Real &energy, Real &virial) const
    {
        unsigned int p = slot(i, j);
        Real sigma = type_sigma[p];
        Real epsilon = type_epsilon[p];
        Real cutoff = type_cutoff[p];
        Real r_switch = type_switch[p];
        Real inv_r2 = R(1.0) / r2;
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
    }

    __device__ inline Real exclusion_scale(unsigned int i, unsigned int j) const {
        return heddle_jit_exclusion_scale(i, j, excl_offsets, excl_partners, excl_scales);
    }
};
"#;
    let entry_point_args = r#"    const unsigned int *lj_type_indices,
    unsigned int lj_n_types,
    const Real *lj_type_sigma,
    const Real *lj_type_epsilon,
    const Real *lj_type_cutoff,
    const Real *lj_type_switch,
    const unsigned int *lj_excl_offsets,
    const unsigned int *lj_excl_partners,
    const Real *lj_excl_scales,
"#;
    let functor_init_source = r#"    composite.functor_lennard_jones.type_indices = lj_type_indices;
    composite.functor_lennard_jones.n_types = lj_n_types;
    composite.functor_lennard_jones.type_sigma = lj_type_sigma;
    composite.functor_lennard_jones.type_epsilon = lj_type_epsilon;
    composite.functor_lennard_jones.type_cutoff = lj_type_cutoff;
    composite.functor_lennard_jones.type_switch = lj_type_switch;
    composite.functor_lennard_jones.excl_offsets = lj_excl_offsets;
    composite.functor_lennard_jones.excl_partners = lj_excl_partners;
    composite.functor_lennard_jones.excl_scales = lj_excl_scales;
"#;
    PairForceFragment {
        label: "lennard_jones",
        functor_struct_name: "LjPairFunctor",
        functor_source: functor_source.to_string(),
        entry_point_args: entry_point_args.to_string(),
        functor_init_source: functor_init_source.to_string(),
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

