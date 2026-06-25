use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, reduce_angle_forces,
};
use crate::kernels;
use crate::io::config::AngleTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::AngleList;
use super::{
    AggregateLevel, AngleForceFragment, AngleScratchView, ForceFieldError, ForceLaunchBuilder,
    ForceLaunchContext, KernelArg, KernelArgBinder, KernelArgSchema, KernelArgType, Potential,
    PotentialBuildContext, PotentialBuilder, SlotOutputView,
};
use crate::precision::Real;

// rq-21a8063c rq-454ad2cf
#[derive(Debug)]
pub struct HarmonicAngleState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub angles: CudaSlice<u32>,
    pub atom_angle_offsets: CudaSlice<u32>,
    pub atom_angle_indices: CudaSlice<u32>,
    pub angle_k_theta: CudaSlice<Real>,
    pub angle_theta_0: CudaSlice<Real>,
    pub angle_triple_x: CudaSlice<Real>,
    pub angle_triple_y: CudaSlice<Real>,
    pub angle_triple_z: CudaSlice<Real>,
    pub angle_triple_energy: CudaSlice<Real>,
    pub angle_triple_virial: CudaSlice<Real>,
    pub angle_count: usize,
    pub particle_count: usize,
}

impl HarmonicAngleState {
    pub fn new(
        gpu: &GpuContext,
        angle_list: &AngleList,
        angle_types: &[AngleTypeConfig],
    ) -> Result<Self, GpuError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let angle_count = angle_list.angles.len();
        let particle_count = angle_list.particle_count;

        // Flatten angles to [atom_i, atom_j, atom_k, type_idx] quadruples.
        let mut angles_flat: Vec<u32> = Vec::with_capacity(4 * angle_count);
        for a in &angle_list.angles {
            angles_flat.push(a.atom_i);
            angles_flat.push(a.atom_j);
            angles_flat.push(a.atom_k);
            angles_flat.push(a.angle_type_index);
        }

        let mut k_vec: Vec<Real> = Vec::with_capacity(angle_types.len());
        let mut theta0_vec: Vec<Real> = Vec::with_capacity(angle_types.len());
        for at in angle_types {
            match at {
                AngleTypeConfig::Harmonic { k_theta, theta_0, .. } => {
                    k_vec.push(*k_theta as Real);
                    theta0_vec.push(*theta_0 as Real);
                }
            }
        }

        let angles = htod_or_empty_u32(&device, &angles_flat)?;
        let atom_angle_offsets = htod_or_empty_u32(&device, &angle_list.atom_angle_offsets)?;
        let atom_angle_indices = htod_or_empty_u32(&device, &angle_list.atom_angle_indices)?;
        let angle_k_theta = htod_or_empty(&device, &k_vec)?;
        let angle_theta_0 = htod_or_empty(&device, &theta0_vec)?;

        let triple_len = 3 * angle_count;
        let angle_triple_x = device.alloc_zeros::<Real>(triple_len).map_err(GpuError::from)?;
        let angle_triple_y = device.alloc_zeros::<Real>(triple_len).map_err(GpuError::from)?;
        let angle_triple_z = device.alloc_zeros::<Real>(triple_len).map_err(GpuError::from)?;
        let angle_triple_energy =
            device.alloc_zeros::<Real>(triple_len).map_err(GpuError::from)?;
        let angle_triple_virial =
            device.alloc_zeros::<Real>(triple_len).map_err(GpuError::from)?;

        Ok(HarmonicAngleState {
            device,
            kernels,
            angles,
            atom_angle_offsets,
            atom_angle_indices,
            angle_k_theta,
            angle_theta_0,
            angle_triple_x,
            angle_triple_y,
            angle_triple_z,
            angle_triple_energy,
            angle_triple_virial,
            angle_count,
            particle_count,
        })
    }
}

impl Potential for HarmonicAngleState {
    fn label(&self) -> &'static str {
        LABEL
    }

    fn max_cutoff(&self) -> Option<Real> {
        None
    }

    fn compute(
        &mut self,
        _buffers: &ParticleBuffers,
        _sim_box: &SimulationBox,
        mut output: SlotOutputView<'_>,
        _cx: &crate::forces::ForceFieldContext<'_>,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 || self.angle_count == 0 {
            // Empty slot is the additive identity; the framework has
            // already prepared the class accumulator.
            return Ok(());
        }
        // The per-angle contribution kernel runs from the framework's
        // JIT-composed angle module dispatch *before* this method; by
        // the time we get here, the slot's angle-triple scratch buffer
        // holds the per-angle contributions. Only the per-atom
        // reduction is the slot's responsibility.
        let write_scalars = matches!(level, AggregateLevel::ForcesAndScalars);
        timings.kernel_start(KernelStage::REDUCE_ANGLE_FORCES)?;
        reduce_angle_forces(
            &self.kernels,
            &self.angle_triple_x,
            &self.angle_triple_y,
            &self.angle_triple_z,
            &self.angle_triple_energy,
            &self.angle_triple_virial,
            &self.atom_angle_offsets,
            &self.atom_angle_indices,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            &mut output.energy,
            &mut output.virial,
            self.particle_count,
            write_scalars,
        )?;
        timings.kernel_stop(KernelStage::REDUCE_ANGLE_FORCES)?;
        Ok(())
    }

    fn bind_angle_force_args(
        &self,
        _ctx: &ForceLaunchContext<'_>,
        builder: &mut ForceLaunchBuilder,
    ) {
        // Validated against `harmonic_angle_arg_schema()` — the same
        // schema that generates the fragment's entry-point args and
        // functor-init source — so the binding cannot drift from the
        // kernel signature.
        let schema = harmonic_angle_arg_schema();
        let mut b = KernelArgBinder::new(&schema, LABEL, builder);
        b.buffer("harmonic_angle_k_theta", &self.angle_k_theta);
        b.buffer("harmonic_angle_theta_0", &self.angle_theta_0);
        b.finish();
    }

    fn angle_scratch(&self) -> Option<AngleScratchView<'_>> {
        Some(AngleScratchView {
            angles: &self.angles,
            angle_triple_x: &self.angle_triple_x,
            angle_triple_y: &self.angle_triple_y,
            angle_triple_z: &self.angle_triple_z,
            angle_triple_energy: &self.angle_triple_energy,
            angle_triple_virial: &self.angle_triple_virial,
            angle_count: self.angle_count,
        })
    }
}

/// The slot's stable label, shared by `Potential::label`, the fragment,
/// and the argument schema.
const LABEL: &str = "harmonic_angle";

/// Single source of truth for the harmonic-angle per-angle kernel
/// arguments. The fragment's `entry_point_args` and `functor_init_source`
/// are generated from this list (local-functor init), and
/// `bind_angle_force_args` is validated against it, so the three pieces
/// cannot drift apart.
fn harmonic_angle_arg_schema() -> KernelArgSchema {
    use KernelArgType::ConstPtrReal;
    KernelArgSchema::intramolecular(
        LABEL,
        vec![
            KernelArg::new("harmonic_angle_k_theta", ConstPtrReal, "angle_k_theta"),
            KernelArg::new("harmonic_angle_theta_0", ConstPtrReal, "angle_theta_0"),
        ],
    )
}

fn htod_or_empty_u32(
    device: &Arc<CudaDevice>,
    data: &[u32],
) -> Result<CudaSlice<u32>, GpuError> {
    if data.is_empty() {
        device.alloc_zeros::<u32>(0).map_err(GpuError::from)
    } else {
        device.htod_sync_copy(data).map_err(GpuError::from)
    }
}

fn htod_or_empty(
    device: &Arc<CudaDevice>,
    data: &[Real],
) -> Result<CudaSlice<Real>, GpuError> {
    if data.is_empty() {
        device.alloc_zeros::<Real>(0).map_err(GpuError::from)
    } else {
        device.htod_sync_copy(data).map_err(GpuError::from)
    }
}

// rq-e8550f96
#[derive(Debug, Clone)]
pub struct HarmonicAngleBuilder;

impl PotentialBuilder for HarmonicAngleBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        if cx.angle_list.is_empty() {
            return Ok(None);
        }
        let state = HarmonicAngleState::new(cx.gpu, cx.angle_list, cx.angle_types)?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }

    fn angle_force_fragment(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<AngleForceFragment>, ForceFieldError> {
        if cx.angle_list.is_empty() {
            return Ok(None);
        }
        Ok(Some(harmonic_angle_force_fragment()))
    }
}

/// Harmonic angle force fragment for the JIT-composed angle module.
/// The functor exposes `evaluate(dx_ij, dy_ij, dz_ij, dx_kj, dy_kj,
/// dz_kj, angle_type_index, fix, fiy, fiz, fkx, fky, fkz, u_m, w_m)`
/// per the contract in `rqm/forces/jit-composed-intramolecular.md`.
pub fn harmonic_angle_force_fragment() -> AngleForceFragment {
    let functor_source = r#"
struct HarmonicAngleFunctor {
    const Real *angle_k_theta;
    const Real *angle_theta_0;

    __device__ inline void evaluate(
        Real dx_ij, Real dy_ij, Real dz_ij,
        Real dx_kj, Real dy_kj, Real dz_kj,
        unsigned int angle_type_index,
        Real &fix, Real &fiy, Real &fiz,
        Real &fkx, Real &fky, Real &fkz,
        Real &u_m,
        Real &w_m) const
    {
        Real dij2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
        Real dkj2 = dx_kj * dx_kj + dy_kj * dy_kj + dz_kj * dz_kj;
        if (dij2 == R(0.0) || dkj2 == R(0.0)) {
            fix = R(0.0); fiy = R(0.0); fiz = R(0.0);
            fkx = R(0.0); fky = R(0.0); fkz = R(0.0);
            u_m = R(0.0); w_m = R(0.0);
            return;
        }
        Real dij = Real_sqrt(dij2);
        Real dkj = Real_sqrt(dkj2);
        Real inv_dij_dkj = R(1.0) / (dij * dkj);
        Real dot = dx_ij * dx_kj + dy_ij * dy_kj + dz_ij * dz_kj;
        Real cos_theta = dot * inv_dij_dkj;
        if (cos_theta >  R(1.0)) cos_theta =  R(1.0);
        if (cos_theta < -R(1.0)) cos_theta = -R(1.0);
        Real sin_sq = R(1.0) - cos_theta * cos_theta;
        Real sin_theta = Real_sqrt(sin_sq > R(0.0) ? sin_sq : R(0.0));
        if (sin_theta < R(1.0e-7)) {
            fix = R(0.0); fiy = R(0.0); fiz = R(0.0);
            fkx = R(0.0); fky = R(0.0); fkz = R(0.0);
            u_m = R(0.0); w_m = R(0.0);
            return;
        }
        Real theta = Real_atan2(dij * dkj * sin_theta, dot);
        Real k = angle_k_theta[angle_type_index];
        Real theta_0 = angle_theta_0[angle_type_index];
        Real dtheta = theta - theta_0;
        Real g = -k * dtheta / sin_theta;
        Real inv_dij2 = R(1.0) / dij2;
        Real inv_dkj2 = R(1.0) / dkj2;
        fix = g * (cos_theta * inv_dij2 * dx_ij - inv_dij_dkj * dx_kj);
        fiy = g * (cos_theta * inv_dij2 * dy_ij - inv_dij_dkj * dy_kj);
        fiz = g * (cos_theta * inv_dij2 * dz_ij - inv_dij_dkj * dz_kj);
        fkx = g * (cos_theta * inv_dkj2 * dx_kj - inv_dij_dkj * dx_ij);
        fky = g * (cos_theta * inv_dkj2 * dy_kj - inv_dij_dkj * dy_ij);
        fkz = g * (cos_theta * inv_dkj2 * dz_kj - inv_dij_dkj * dz_ij);
        u_m = R(0.5) * k * dtheta * dtheta;
        w_m = (dx_ij * fix + dy_ij * fiy + dz_ij * fiz)
            + (dx_kj * fkx + dy_kj * fky + dz_kj * fkz);
    }
};
"#;
    // `entry_point_args` and `functor_init_source` are generated from
    // `harmonic_angle_arg_schema()`, the same schema
    // `bind_angle_force_args` is validated against; the functor field
    // names in `functor_source` above must match the schema's
    // `functor_field` entries.
    let schema = harmonic_angle_arg_schema();
    AngleForceFragment {
        label: LABEL,
        functor_struct_name: "HarmonicAngleFunctor",
        functor_source: functor_source.to_string(),
        entry_point_args: schema.entry_point_args(),
        functor_init_source: schema.functor_init_source(),
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct AngleKernels {
    pub reduce_angle_forces: CudaFunction,
}

impl AngleKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::ANGLE),
            "angle",
            &["reduce_angle_forces"],
        )?;
        Ok(AngleKernels {
            reduce_angle_forces: get_func(device, "angle", "reduce_angle_forces")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The CUDA argument declarations and local-functor initialisation
    // the angle composer expects for the HarmonicAngle slot.
    const EXPECTED_ENTRY_POINT_ARGS: &str = r#"    const Real *harmonic_angle_k_theta,
    const Real *harmonic_angle_theta_0,
"#;

    const EXPECTED_FUNCTOR_INIT_SOURCE: &str = r#"    functor.angle_k_theta = harmonic_angle_k_theta;
    functor.angle_theta_0 = harmonic_angle_theta_0;
"#;

    #[test]
    fn generated_entry_point_args_match_expected() {
        assert_eq!(
            harmonic_angle_arg_schema().entry_point_args(),
            EXPECTED_ENTRY_POINT_ARGS
        );
    }

    #[test]
    fn generated_functor_init_source_is_local_functor() {
        let init = harmonic_angle_arg_schema().functor_init_source();
        assert_eq!(init, EXPECTED_FUNCTOR_INIT_SOURCE);
        assert!(!init.contains("composite."));
    }
}
