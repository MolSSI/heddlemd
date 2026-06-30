use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, reduce_dihedral_forces,
};
use crate::io::config::DihedralTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::{Dihedral, DihedralList};
use super::{
    AggregateLevel, DihedralForceFragment, DihedralPotential, DihedralScratchView,
    ForceFieldError, ForceLaunchBuilder, ForceLaunchContext, JitParticipant, KernelArg,
    KernelArgBinder, KernelArgSchema, KernelArgType, Potential, PotentialBuildContext,
    PotentialBuilder, SlotOutputView,
};
use crate::precision::Real;

// rq-4b84f452 rq-ccea967a
/// Periodic-dihedral potential slot. Evaluates
/// `U(φ) = k_phi · (1 + cos(n · φ − phi_0))` per dihedral. Only the
/// subset of `DihedralList` entries whose `dihedral_type_index`
/// references a `DihedralTypeConfig::Periodic` entry is uploaded to
/// this slot; entries of other functional forms live in their own
/// slot driven by the same `DihedralList`.
#[derive(Debug)]
pub struct PeriodicDihedralState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    /// `[atom_i, atom_j, atom_k, atom_l, periodic_type_index]`
    /// quintuples, length `5 · D` where `D` is the count of dihedrals
    /// referencing periodic types.
    pub dihedrals: CudaSlice<u32>,
    pub atom_dihedral_offsets: CudaSlice<u32>,
    pub atom_dihedral_indices: CudaSlice<u32>,
    pub dihedral_k_phi: CudaSlice<Real>,
    pub dihedral_phi_0: CudaSlice<Real>,
    pub dihedral_n: CudaSlice<u32>,
    pub dihedral_quadruple_x: CudaSlice<Real>,
    pub dihedral_quadruple_y: CudaSlice<Real>,
    pub dihedral_quadruple_z: CudaSlice<Real>,
    pub dihedral_quadruple_energy: CudaSlice<Real>,
    pub dihedral_quadruple_virial: CudaSlice<Real>,
    pub dihedral_count: usize,
    pub particle_count: usize,
}

impl PeriodicDihedralState {
    pub fn new(
        gpu: &GpuContext,
        dihedral_list: &DihedralList,
        dihedral_types: &[DihedralTypeConfig],
    ) -> Result<Self, GpuError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let particle_count = dihedral_list.particle_count;

        // Build a global -> periodic type index remap so the device-
        // side `dihedrals` array references the compact periodic-only
        // parameter tables. `None` means the global type is not a
        // periodic type and any dihedral that references it is
        // filtered out.
        let mut periodic_remap: Vec<Option<u32>> = vec![None; dihedral_types.len()];
        let mut k_vec: Vec<Real> = Vec::new();
        let mut phi0_vec: Vec<Real> = Vec::new();
        let mut n_vec: Vec<u32> = Vec::new();
        for (global_idx, dt) in dihedral_types.iter().enumerate() {
            match dt {
                DihedralTypeConfig::Periodic {
                    k_phi, n, phi_0, ..
                } => {
                    periodic_remap[global_idx] = Some(k_vec.len() as u32);
                    k_vec.push(*k_phi as Real);
                    phi0_vec.push(*phi_0 as Real);
                    n_vec.push(*n);
                }
            }
        }

        // Filter the dihedral list to entries with periodic types
        // and rewrite the dihedral_type_index to the periodic-only
        // table. Preserve canonical order.
        let mut filtered: Vec<Dihedral> = Vec::new();
        let mut original_to_filtered: Vec<Option<u32>> =
            vec![None; dihedral_list.dihedrals.len()];
        for (orig_idx, d) in dihedral_list.dihedrals.iter().enumerate() {
            let global = d.dihedral_type_index as usize;
            if let Some(local) = periodic_remap[global] {
                original_to_filtered[orig_idx] = Some(filtered.len() as u32);
                filtered.push(Dihedral {
                    atom_i: d.atom_i,
                    atom_j: d.atom_j,
                    atom_k: d.atom_k,
                    atom_l: d.atom_l,
                    dihedral_type_index: local,
                });
            }
        }
        let dihedral_count = filtered.len();

        // Flatten to [atom_i, atom_j, atom_k, atom_l, type_idx]
        // quintuples for the device. Layout matches the JIT outer-loop
        // body in `jit_composed.rs::emit_dihedral_entry_point` (5 u32s
        // per dihedral).
        let mut dihedrals_flat: Vec<u32> = Vec::with_capacity(5 * dihedral_count);
        for d in &filtered {
            dihedrals_flat.push(d.atom_i);
            dihedrals_flat.push(d.atom_j);
            dihedrals_flat.push(d.atom_k);
            dihedrals_flat.push(d.atom_l);
            dihedrals_flat.push(d.dihedral_type_index);
        }

        // Rebuild atom_dihedral_offsets / atom_dihedral_indices from
        // the filtered subset so per-atom indexing is internally
        // consistent. (The DihedralList's tables are keyed off the
        // unfiltered ordering; rebuilding here keeps the reduction
        // kernel reading the right scratch slots.)
        let mut atom_dihedral_offsets = vec![0u32; particle_count + 1];
        for d in &filtered {
            atom_dihedral_offsets[d.atom_i as usize + 1] += 1;
            atom_dihedral_offsets[d.atom_j as usize + 1] += 1;
            atom_dihedral_offsets[d.atom_k as usize + 1] += 1;
            atom_dihedral_offsets[d.atom_l as usize + 1] += 1;
        }
        for i in 1..=particle_count {
            atom_dihedral_offsets[i] += atom_dihedral_offsets[i - 1];
        }
        let mut atom_dihedral_indices = vec![0u32; dihedral_count * 4];
        let mut cursor: Vec<u32> = atom_dihedral_offsets[..particle_count].to_vec();
        for (m, d) in filtered.iter().enumerate() {
            let slot_i = (4 * m) as u32;
            let slot_j = (4 * m + 1) as u32;
            let slot_k = (4 * m + 2) as u32;
            let slot_l = (4 * m + 3) as u32;
            let pi = d.atom_i as usize;
            let pj = d.atom_j as usize;
            let pk = d.atom_k as usize;
            let pl = d.atom_l as usize;
            atom_dihedral_indices[cursor[pi] as usize] = slot_i;
            cursor[pi] += 1;
            atom_dihedral_indices[cursor[pj] as usize] = slot_j;
            cursor[pj] += 1;
            atom_dihedral_indices[cursor[pk] as usize] = slot_k;
            cursor[pk] += 1;
            atom_dihedral_indices[cursor[pl] as usize] = slot_l;
            cursor[pl] += 1;
        }

        let dihedrals_buf = htod_or_empty_u32(&device, &dihedrals_flat)?;
        let atom_dihedral_offsets_buf =
            htod_or_empty_u32(&device, &atom_dihedral_offsets)?;
        let atom_dihedral_indices_buf =
            htod_or_empty_u32(&device, &atom_dihedral_indices)?;
        let dihedral_k_phi = htod_or_empty(&device, &k_vec)?;
        let dihedral_phi_0 = htod_or_empty(&device, &phi0_vec)?;
        let dihedral_n = htod_or_empty_u32(&device, &n_vec)?;

        let quad_len = 4 * dihedral_count;
        let dihedral_quadruple_x =
            device.alloc_zeros::<Real>(quad_len).map_err(GpuError::from)?;
        let dihedral_quadruple_y =
            device.alloc_zeros::<Real>(quad_len).map_err(GpuError::from)?;
        let dihedral_quadruple_z =
            device.alloc_zeros::<Real>(quad_len).map_err(GpuError::from)?;
        let dihedral_quadruple_energy =
            device.alloc_zeros::<Real>(quad_len).map_err(GpuError::from)?;
        let dihedral_quadruple_virial =
            device.alloc_zeros::<Real>(quad_len).map_err(GpuError::from)?;

        Ok(PeriodicDihedralState {
            device,
            kernels,
            dihedrals: dihedrals_buf,
            atom_dihedral_offsets: atom_dihedral_offsets_buf,
            atom_dihedral_indices: atom_dihedral_indices_buf,
            dihedral_k_phi,
            dihedral_phi_0,
            dihedral_n,
            dihedral_quadruple_x,
            dihedral_quadruple_y,
            dihedral_quadruple_z,
            dihedral_quadruple_energy,
            dihedral_quadruple_virial,
            dihedral_count,
            particle_count,
        })
    }
}

impl Potential for PeriodicDihedralState {
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
        if self.particle_count == 0 || self.dihedral_count == 0 {
            // Empty slot is the additive identity; the framework has
            // already prepared the class accumulator.
            return Ok(());
        }
        // The per-dihedral contribution kernel runs from the
        // framework's JIT-composed dihedral module dispatch *before*
        // this method; by the time we get here, the slot's
        // dihedral-quadruple scratch buffer holds the per-dihedral
        // contributions. Only the per-atom reduction is the slot's
        // responsibility.
        let write_scalars = matches!(level, AggregateLevel::ForcesAndScalars);
        timings.kernel_start(KernelStage::REDUCE_DIHEDRAL_FORCES)?;
        reduce_dihedral_forces(
            &self.kernels,
            &self.dihedral_quadruple_x,
            &self.dihedral_quadruple_y,
            &self.dihedral_quadruple_z,
            &self.dihedral_quadruple_energy,
            &self.dihedral_quadruple_virial,
            &self.atom_dihedral_offsets,
            &self.atom_dihedral_indices,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            &mut output.energy,
            &mut output.virial,
            self.particle_count,
            write_scalars,
        )?;
        timings.kernel_stop(KernelStage::REDUCE_DIHEDRAL_FORCES)?;
        Ok(())
    }

    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        Some(JitParticipant::Dihedral(self))
    }
}

impl DihedralPotential for PeriodicDihedralState {
    fn dihedral_force_fragment(&self) -> DihedralForceFragment {
        periodic_dihedral_force_fragment()
    }

    fn dihedral_scratch(&self) -> DihedralScratchView<'_> {
        DihedralScratchView {
            dihedrals: &self.dihedrals,
            dihedral_quadruple_x: &self.dihedral_quadruple_x,
            dihedral_quadruple_y: &self.dihedral_quadruple_y,
            dihedral_quadruple_z: &self.dihedral_quadruple_z,
            dihedral_quadruple_energy: &self.dihedral_quadruple_energy,
            dihedral_quadruple_virial: &self.dihedral_quadruple_virial,
            dihedral_count: self.dihedral_count,
        }
    }

    fn bind_dihedral_force_args(
        &self,
        _ctx: &ForceLaunchContext<'_>,
        builder: &mut ForceLaunchBuilder,
    ) {
        // Validated against `periodic_dihedral_arg_schema()` — the same
        // schema that generates the fragment's entry-point args and
        // functor-init source — so the binding cannot drift from the
        // kernel signature.
        let schema = periodic_dihedral_arg_schema();
        let mut b = KernelArgBinder::new(&schema, LABEL, builder);
        b.buffer("periodic_dihedral_k_phi", &self.dihedral_k_phi);
        b.buffer("periodic_dihedral_phi_0", &self.dihedral_phi_0);
        b.buffer("periodic_dihedral_n", &self.dihedral_n);
        b.finish();
    }
}

/// The slot's stable label, shared by `Potential::label`, the fragment,
/// and the argument schema.
const LABEL: &str = "periodic_dihedral";

/// Single source of truth for the periodic-dihedral per-dihedral
/// kernel arguments.
fn periodic_dihedral_arg_schema() -> KernelArgSchema {
    use KernelArgType::{ConstPtrReal, ConstPtrU32};
    KernelArgSchema::intramolecular(
        LABEL,
        vec![
            KernelArg::new("periodic_dihedral_k_phi", ConstPtrReal, "dihedral_k_phi"),
            KernelArg::new("periodic_dihedral_phi_0", ConstPtrReal, "dihedral_phi_0"),
            KernelArg::new("periodic_dihedral_n", ConstPtrU32, "dihedral_n"),
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

#[derive(Debug, Clone)]
pub struct PeriodicDihedralBuilder;

impl PotentialBuilder for PeriodicDihedralBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        if cx.dihedral_list.is_empty() {
            return Ok(None);
        }
        // The slot only activates when at least one entry of the
        // dihedral list resolves to a periodic type — there is no
        // point compiling a JIT module that processes zero dihedrals.
        let any_periodic = cx.dihedral_list.dihedrals.iter().any(|d| {
            matches!(
                cx.dihedral_types.get(d.dihedral_type_index as usize),
                Some(DihedralTypeConfig::Periodic { .. })
            )
        });
        if !any_periodic {
            return Ok(None);
        }
        let state = PeriodicDihedralState::new(cx.gpu, cx.dihedral_list, cx.dihedral_types)?;
        Ok(Some(Box::new(state)))
    }
}

// rq-9fd47b06
/// Periodic dihedral force fragment for the JIT-composed dihedral
/// module. The functor implements the per-dihedral chain-rule
/// derivation given in `rqm/forces/periodic-dihedral.md`'s *Algorithm*
/// section.
pub fn periodic_dihedral_force_fragment() -> DihedralForceFragment {
    // Geometry / force math: standard formulation (matches GROMACS /
    // OpenMM / CHARMM). See `periodic-dihedral.md`. Note the
    // convention `b3 = r_k − r_l` (not `r_l − r_k`); the standard
    // cross-product formula yields the canonical IUPAC torsion only
    // with this sign of `b3`.
    //
    //   b1 = r_i − r_j           (passed in as dx_ij, dy_ij, dz_ij)
    //   b2 = r_k − r_j           (passed in as dx_kj, dy_kj, dz_kj)
    //   b3 = r_k − r_l           (passed in as dx_kl, dy_kl, dz_kl)
    //   m  = b1 × b2,  n_vec = b2 × b3
    //   φ  = atan2(|b2| · (b1 · n_vec),  m · n_vec)
    //   f_φ = k · n · sin(n φ − phi_0)            (= −dU/dφ)
    //   F_i = f_φ · ( |b2| / |m|² ) · m
    //   F_l = f_φ · (−|b2| / |n_vec|²) · n_vec
    //   s   = (b1 · b2) / |b2|²,   t = (b3 · b2) / |b2|²
    //   F_j = (s − 1) · F_i  −  t       · F_l
    //   F_k = (−s)    · F_i  +  (t − 1) · F_l
    //
    // Newton's third law gives F_i + F_j + F_k + F_l = 0 to within
    // f32 round-off.
    //
    // The functor's defensive guard zeros every output when any of
    // |m|², |n_vec|², |b2|² falls below 1e-14 (degenerate geometry).
    let functor_source = r#"
struct PeriodicDihedralFunctor {
    const Real *dihedral_k_phi;
    const Real *dihedral_phi_0;
    const unsigned int *dihedral_n;

    __device__ inline void evaluate(
        Real dx_ij, Real dy_ij, Real dz_ij,
        Real dx_kj, Real dy_kj, Real dz_kj,
        Real dx_kl, Real dy_kl, Real dz_kl,
        unsigned int dihedral_type_index,
        Real &fix, Real &fiy, Real &fiz,
        Real &fjx, Real &fjy, Real &fjz,
        Real &fkx, Real &fky, Real &fkz,
        Real &flx, Real &fly, Real &flz,
        Real &u_m,
        Real &w_m) const
    {
        // m = b1 x b2, n_vec = b2 x b3 with b3 = r_k - r_l (dx_kl).
        Real mx = dy_ij * dz_kj - dz_ij * dy_kj;
        Real my = dz_ij * dx_kj - dx_ij * dz_kj;
        Real mz = dx_ij * dy_kj - dy_ij * dx_kj;
        Real nx = dy_kj * dz_kl - dz_kj * dy_kl;
        Real ny = dz_kj * dx_kl - dx_kj * dz_kl;
        Real nz = dx_kj * dy_kl - dy_kj * dx_kl;

        Real m2 = mx * mx + my * my + mz * mz;
        Real n2 = nx * nx + ny * ny + nz * nz;
        Real b2sq = dx_kj * dx_kj + dy_kj * dy_kj + dz_kj * dz_kj;
        if (m2 < R(1.0e-14) || n2 < R(1.0e-14) || b2sq < R(1.0e-14)) {
            fix = R(0.0); fiy = R(0.0); fiz = R(0.0);
            fjx = R(0.0); fjy = R(0.0); fjz = R(0.0);
            fkx = R(0.0); fky = R(0.0); fkz = R(0.0);
            flx = R(0.0); fly = R(0.0); flz = R(0.0);
            u_m = R(0.0); w_m = R(0.0);
            return;
        }

        Real b2_len = Real_sqrt(b2sq);
        // sin component (sign-determining): b1 · n_vec, scaled by |b2|.
        Real b1_dot_n = dx_ij * nx + dy_ij * ny + dz_ij * nz;
        Real m_dot_n  = mx * nx + my * ny + mz * nz;
        Real phi = Real_atan2(b2_len * b1_dot_n, m_dot_n);

        Real k = dihedral_k_phi[dihedral_type_index];
        Real phi_0 = dihedral_phi_0[dihedral_type_index];
        unsigned int n_mult = dihedral_n[dihedral_type_index];
        Real n_real = (Real) n_mult;
        Real delta = n_real * phi - phi_0;
        Real sin_d = Real_sin(delta);
        Real cos_d = Real_cos(delta);
        // f_phi = -dU/dphi = k · n · sin(n φ − phi_0)
        Real f_phi = k * n_real * sin_d;

        // ∂φ/∂r_i and ∂φ/∂r_l
        Real coef_i = f_phi * b2_len / m2;
        Real coef_l = -f_phi * b2_len / n2;
        fix = coef_i * mx;
        fiy = coef_i * my;
        fiz = coef_i * mz;
        flx = coef_l * nx;
        fly = coef_l * ny;
        flz = coef_l * nz;

        Real inv_b2sq = R(1.0) / b2sq;
        Real s = (dx_ij * dx_kj + dy_ij * dy_kj + dz_ij * dz_kj) * inv_b2sq;
        Real t = (dx_kl * dx_kj + dy_kl * dy_kj + dz_kl * dz_kj) * inv_b2sq;
        fjx = (s - R(1.0)) * fix - t * flx;
        fjy = (s - R(1.0)) * fiy - t * fly;
        fjz = (s - R(1.0)) * fiz - t * flz;
        fkx = -s * fix + (t - R(1.0)) * flx;
        fky = -s * fiy + (t - R(1.0)) * fly;
        fkz = -s * fiz + (t - R(1.0)) * flz;

        u_m = k * (R(1.0) + cos_d);
        // Virial with j as reference: W = Σ_a (r_a − r_j)·F_a, i.e.
        //   W = b1·F_i + 0·F_j + b2·F_k + (b2 − b3)·F_l
        // (where r_l − r_j = (r_l − r_k) + (r_k − r_j) = −b3 + b2).
        Real b2_minus_b3_x = dx_kj - dx_kl;
        Real b2_minus_b3_y = dy_kj - dy_kl;
        Real b2_minus_b3_z = dz_kj - dz_kl;
        w_m = (dx_ij * fix + dy_ij * fiy + dz_ij * fiz)
            + (dx_kj * fkx + dy_kj * fky + dz_kj * fkz)
            + (b2_minus_b3_x * flx + b2_minus_b3_y * fly + b2_minus_b3_z * flz);
    }
};
"#;
    let schema = periodic_dihedral_arg_schema();
    DihedralForceFragment {
        label: LABEL,
        functor_struct_name: "PeriodicDihedralFunctor",
        functor_source: functor_source.to_string(),
        entry_point_args: schema.entry_point_args(),
        functor_init_source: schema.functor_init_source(),
    }
}

// rq-fb1676f8 rq-2932ea42
crate::gpu_kernels! {
    module: "dihedral",
    ptx: crate::kernels::DIHEDRAL,
    struct: DihedralKernels,
    kernels: [reduce_dihedral_forces],
    stages: {
        REDUCE_DIHEDRAL_FORCES = "reduce_dihedral_forces",
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPECTED_ENTRY_POINT_ARGS: &str = r#"    const Real *periodic_dihedral_k_phi,
    const Real *periodic_dihedral_phi_0,
    const unsigned int *periodic_dihedral_n,
"#;

    const EXPECTED_FUNCTOR_INIT_SOURCE: &str = r#"    functor.dihedral_k_phi = periodic_dihedral_k_phi;
    functor.dihedral_phi_0 = periodic_dihedral_phi_0;
    functor.dihedral_n = periodic_dihedral_n;
"#;

    #[test]
    fn generated_entry_point_args_match_expected() {
        assert_eq!(
            periodic_dihedral_arg_schema().entry_point_args(),
            EXPECTED_ENTRY_POINT_ARGS
        );
    }

    #[test]
    fn generated_functor_init_source_is_local_functor() {
        let init = periodic_dihedral_arg_schema().functor_init_source();
        assert_eq!(init, EXPECTED_FUNCTOR_INIT_SOURCE);
        assert!(!init.contains("composite."));
    }
}
