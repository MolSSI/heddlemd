use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, reduce_angle_forces,
};
use crate::io::config::AngleTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::AngleList;
use super::morse::require_finite_positive;
use super::{
    AggregateLevel, AngleForceFragment, AnglePotential, AngleScratchView, ForceFieldError,
    ForceLaunchBuilder, ForceLaunchContext, JitParticipant, KernelArg, KernelArgBinder,
    KernelArgSchema, KernelArgType, Potential, PotentialBuildContext, PotentialBuilder,
    PotentialConfigEntry, PotentialParamsCategory, PotentialParamsClaim, SlotOutputView,
};
use crate::io::config::{ConfigError, translate_params_error_local};
use crate::precision::Real;

/// The `kind` string the harmonic-angle builder claims in
/// `[[angle_types]]`.
pub const HARMONIC_ANGLE_KIND: &str = "harmonic";

// rq-b33243ff
/// Typed per-entry parameter struct the harmonic-angle builder
/// deserialises from a claimed `kind = "harmonic"` `[[angle_types]]`
/// entry's `params`. `theta_0` is dimensionless for unit conversion
/// (radians in both unit systems).
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct HarmonicAngleParams {
    pub k_theta: crate::units::Energy,
    pub theta_0: f64,
}

fn is_harmonic(angle_types: &[AngleTypeConfig], ti: u32) -> bool {
    angle_types
        .get(ti as usize)
        .is_some_and(|at| at.kind == HARMONIC_ANGLE_KIND)
}

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
        let particle_count = angle_list.particle_count;

        // Select only the angles whose type this slot claims
        // (`kind = "harmonic"`), preserving the AngleList's sort order.
        // The parameter table stays addressed by the global
        // `angle_type_index`; rows for other kinds hold placeholders
        // this slot never reads.
        let selected: Vec<&super::topology::Angle> = angle_list
            .angles
            .iter()
            .filter(|a| is_harmonic(angle_types, a.angle_type_index))
            .collect();
        let angle_count = selected.len();

        // Flatten angles to [atom_i, atom_j, atom_k, type_idx] quadruples.
        let mut angles_flat: Vec<u32> = Vec::with_capacity(4 * angle_count);
        for a in &selected {
            angles_flat.push(a.atom_i);
            angles_flat.push(a.atom_j);
            angles_flat.push(a.atom_k);
            angles_flat.push(a.angle_type_index);
        }

        let mut k_vec: Vec<Real> = Vec::with_capacity(angle_types.len());
        let mut theta0_vec: Vec<Real> = Vec::with_capacity(angle_types.len());
        for at in angle_types {
            if at.kind == HARMONIC_ANGLE_KIND {
                // rq-b33243ff — validation has already guaranteed the
                // params shape; a failure here is an internal error.
                let p: HarmonicAngleParams = at
                    .params
                    .clone()
                    .try_into()
                    .expect("validated harmonic angle-type params (config invariant)");
                k_vec.push(p.k_theta.0 as Real);
                theta0_vec.push(p.theta_0 as Real);
            } else {
                k_vec.push(0.0);
                theta0_vec.push(0.0);
            }
        }

        // Rebuild atom_angle_offsets / atom_angle_indices over the
        // selected subset so the reduction kernel's per-atom indexing
        // matches the slot's own scratch layout (three slots per
        // selected angle: 3·k, 3·k+1, 3·k+2). Mirrors the shared
        // AngleList map construction in `topology.rs`, restricted to
        // the subset.
        let mut offsets_host = vec![0u32; particle_count + 1];
        for a in &selected {
            offsets_host[a.atom_i as usize + 1] += 1;
            offsets_host[a.atom_j as usize + 1] += 1;
            offsets_host[a.atom_k as usize + 1] += 1;
        }
        for i in 1..=particle_count {
            offsets_host[i] += offsets_host[i - 1];
        }
        let mut indices_host = vec![0u32; angle_count * 3];
        let mut cursor: Vec<u32> = offsets_host[..particle_count].to_vec();
        for (k, a) in selected.iter().enumerate() {
            let slots = [(3 * k) as u32, (3 * k + 1) as u32, (3 * k + 2) as u32];
            for (slot, atom) in slots
                .iter()
                .zip([a.atom_i as usize, a.atom_j as usize, a.atom_k as usize])
            {
                indices_host[cursor[atom] as usize] = *slot;
                cursor[atom] += 1;
            }
        }

        let angles = htod_or_empty_u32(&device, &angles_flat)?;
        let atom_angle_offsets = htod_or_empty_u32(&device, &offsets_host)?;
        let atom_angle_indices = htod_or_empty_u32(&device, &indices_host)?;
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

    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        Some(JitParticipant::Angle(self))
    }
}

impl AnglePotential for HarmonicAngleState {
    fn angle_force_fragment(&self) -> AngleForceFragment {
        harmonic_angle_force_fragment()
    }

    fn angle_scratch(&self) -> AngleScratchView<'_> {
        AngleScratchView {
            angles: &self.angles,
            angle_triple_x: &self.angle_triple_x,
            angle_triple_y: &self.angle_triple_y,
            angle_triple_z: &self.angle_triple_z,
            angle_triple_energy: &self.angle_triple_energy,
            angle_triple_virial: &self.angle_triple_virial,
            angle_count: self.angle_count,
        }
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
        // Active only when at least one angle uses a harmonic angle type.
        let has_harmonic = cx
            .angle_list
            .angles
            .iter()
            .any(|a| is_harmonic(cx.angle_types, a.angle_type_index));
        if !has_harmonic {
            return Ok(None);
        }
        let state = HarmonicAngleState::new(cx.gpu, cx.angle_list, cx.angle_types)?;
        Ok(Some(Box::new(state)))
    }

    // rq-529b9e4b rq-35a89768
    fn params_claim(&self) -> Option<PotentialParamsClaim> {
        Some(PotentialParamsClaim {
            category: PotentialParamsCategory::AngleType,
            kind: HARMONIC_ANGLE_KIND,
        })
    }

    // rq-b33243ff rq-95a50109 — k_theta required, finite, strictly
    // positive; theta_0 required, finite, in [0, π]. Errors carry
    // entry-relative paths; the loader prefixes the document location.
    fn validate_params(
        &self,
        entry: PotentialConfigEntry<'_>,
    ) -> Result<(), ConfigError> {
        let PotentialConfigEntry::AngleType(at) = entry else {
            unreachable!("dispatch guarantees the claimed category");
        };
        let p: HarmonicAngleParams = at
            .params
            .clone()
            .try_into()
            .map_err(translate_params_error_local)?;
        require_finite_positive("k_theta", p.k_theta.0)?;
        if !p.theta_0.is_finite() || !(0.0..=std::f64::consts::PI).contains(&p.theta_0) {
            return Err(ConfigError::InvalidValue {
                field: "theta_0".to_string(),
                reason: "theta_0 must be finite and in [0, π]".to_string(),
            });
        }
        Ok(())
    }

    // rq-529b9e4b
    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), ConfigError> {
        crate::registry::convert_params_in_place::<HarmonicAngleParams>(units, params)
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
crate::gpu_kernels! {
    module: "angle",
    ptx: crate::kernels::ANGLE,
    struct: AngleKernels,
    kernels: [reduce_angle_forces],
    stages: {
        REDUCE_ANGLE_FORCES = "reduce_angle_forces",
    },
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
