//! Harmonic (Hooke's-law) bonded potential slot.
//!
//! Parallels the Morse slot (`morse.rs`): the per-bond contribution is a
//! functor composed into the JIT bonded module, and the per-atom
//! reduction reuses the shape-universal `reduce_bond_forces` kernel. See
//! `rqm/forces/harmonic-bond.md`.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{GpuContext, GpuError, Kernels, ParticleBuffers, reduce_bond_forces};
use crate::io::config::BondTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::BondList;
use super::{
    AggregateLevel, BondedForceFragment, BondedPotential, BondedScratchView, ForceFieldError,
    ForceLaunchBuilder, ForceLaunchContext, JitParticipant, KernelArg, KernelArgBinder,
    KernelArgSchema, KernelArgType, Potential, PotentialBuildContext, PotentialBuilder,
    SlotOutputView,
};
use crate::precision::Real;

// rq-c3da9ee1
#[derive(Debug)]
pub struct HarmonicBondState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub bonds: CudaSlice<u32>,
    pub atom_bond_offsets: CudaSlice<u32>,
    pub atom_bond_indices: CudaSlice<u32>,
    pub bond_k: CudaSlice<Real>,
    pub bond_r0: CudaSlice<Real>,
    pub bond_pair_x: CudaSlice<Real>,
    pub bond_pair_y: CudaSlice<Real>,
    pub bond_pair_z: CudaSlice<Real>,
    pub bond_pair_energy: CudaSlice<Real>,
    pub bond_pair_virial: CudaSlice<Real>,
    pub bond_count: usize,
    pub particle_count: usize,
}

impl HarmonicBondState {
    pub fn new(
        gpu: &GpuContext,
        bond_list: &BondList,
        bond_types: &[BondTypeConfig],
    ) -> Result<Self, GpuError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let particle_count = bond_list.particle_count;

        // rq-f62d94d2 — select only the bonds whose type is harmonic and
        // rebuild this slot's own reduction map over that subset.
        let harmonic_list =
            bond_list.filter_by_type_index(|ti| is_harmonic(bond_types, ti));
        let bond_count = harmonic_list.bonds.len();

        let mut bonds_flat: Vec<u32> = Vec::with_capacity(3 * bond_count);
        for b in &harmonic_list.bonds {
            bonds_flat.push(b.atom_i);
            bonds_flat.push(b.atom_j);
            bonds_flat.push(b.bond_type_index);
        }

        // rq-4943810f — the parameter table is addressed by the global
        // `bond_type_index` and is therefore sized to the full
        // `[[bond_types]]` array; rows for non-harmonic types hold
        // placeholders this slot never reads. `k` is stored directly (not
        // `k/2`): the analytic derivative absorbs the ½ convention, so the
        // force path carries no half-factor (see *Prefactor collapse*).
        let mut k_vec: Vec<Real> = Vec::with_capacity(bond_types.len());
        let mut r0_vec: Vec<Real> = Vec::with_capacity(bond_types.len());
        for bt in bond_types {
            match bt {
                BondTypeConfig::Harmonic { k, r0, .. } => {
                    k_vec.push(*k as Real);
                    r0_vec.push(*r0 as Real);
                }
                BondTypeConfig::Morse { .. } => {
                    k_vec.push(0.0);
                    r0_vec.push(0.0);
                }
            }
        }

        let bonds = htod_or_empty_u32(&device, &bonds_flat)?;
        let atom_bond_offsets = htod_or_empty_u32(&device, &harmonic_list.atom_bond_offsets)?;
        let atom_bond_indices = htod_or_empty_u32(&device, &harmonic_list.atom_bond_indices)?;
        let bond_k = htod_or_empty(&device, &k_vec)?;
        let bond_r0 = htod_or_empty(&device, &r0_vec)?;

        let bond_pair_len = 2 * bond_count;
        let bond_pair_x = device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_y = device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_z = device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_energy =
            device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_virial =
            device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;

        Ok(HarmonicBondState {
            device,
            kernels,
            bonds,
            atom_bond_offsets,
            atom_bond_indices,
            bond_k,
            bond_r0,
            bond_pair_x,
            bond_pair_y,
            bond_pair_z,
            bond_pair_energy,
            bond_pair_virial,
            bond_count,
            particle_count,
        })
    }
}

impl Potential for HarmonicBondState {
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
        if self.particle_count == 0 || self.bond_count == 0 {
            // Empty slot is the additive identity; the framework has
            // already prepared the class accumulator.
            return Ok(());
        }
        // The per-bond contribution kernel runs from the framework's
        // JIT-composed bonded module dispatch *before* this method; by
        // the time we get here, the slot's bond-pair scratch buffer holds
        // the per-bond contributions. Only the per-atom reduction is the
        // slot's responsibility, and it reuses the universal kernel.
        let write_scalars = matches!(level, AggregateLevel::ForcesAndScalars);
        timings.kernel_start(KernelStage::REDUCE_BOND_FORCES)?;
        reduce_bond_forces(
            &self.kernels,
            &self.bond_pair_x,
            &self.bond_pair_y,
            &self.bond_pair_z,
            &self.bond_pair_energy,
            &self.bond_pair_virial,
            &self.atom_bond_offsets,
            &self.atom_bond_indices,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            &mut output.energy,
            &mut output.virial,
            self.particle_count,
            write_scalars,
        )?;
        timings.kernel_stop(KernelStage::REDUCE_BOND_FORCES)?;
        Ok(())
    }

    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        Some(JitParticipant::Bonded(self))
    }
}

impl BondedPotential for HarmonicBondState {
    fn bonded_force_fragment(&self) -> BondedForceFragment {
        harmonic_bonded_force_fragment()
    }

    fn bonded_scratch(&self) -> BondedScratchView<'_> {
        BondedScratchView {
            bonds: &self.bonds,
            bond_pair_x: &self.bond_pair_x,
            bond_pair_y: &self.bond_pair_y,
            bond_pair_z: &self.bond_pair_z,
            bond_pair_energy: &self.bond_pair_energy,
            bond_pair_virial: &self.bond_pair_virial,
            bond_count: self.bond_count,
        }
    }

    fn bind_bonded_force_args(
        &self,
        _ctx: &ForceLaunchContext<'_>,
        builder: &mut ForceLaunchBuilder,
    ) {
        // Validated against `harmonic_arg_schema()` — the same schema that
        // generates the fragment's entry-point args and functor-init
        // source — so the binding cannot drift from the kernel signature.
        let schema = harmonic_arg_schema();
        let mut b = KernelArgBinder::new(&schema, LABEL, builder);
        b.buffer("harmonic_bond_k", &self.bond_k);
        b.buffer("harmonic_bond_r0", &self.bond_r0);
        b.finish();
    }
}

/// The slot's stable label, shared by `Potential::label`, the fragment,
/// and the argument schema.
const LABEL: &str = "harmonic_bond";

/// Single source of truth for the harmonic per-bond kernel arguments. The
/// fragment's `entry_point_args` and `functor_init_source` are generated
/// from this list, and `bind_bonded_force_args` is validated against it,
/// so the three pieces cannot drift apart.
fn harmonic_arg_schema() -> KernelArgSchema {
    use KernelArgType::ConstPtrReal;
    KernelArgSchema::intramolecular(
        LABEL,
        vec![
            KernelArg::new("harmonic_bond_k", ConstPtrReal, "bond_k"),
            KernelArg::new("harmonic_bond_r0", ConstPtrReal, "bond_r0"),
        ],
    )
}

fn is_harmonic(bond_types: &[BondTypeConfig], ti: u32) -> bool {
    matches!(
        bond_types.get(ti as usize),
        Some(BondTypeConfig::Harmonic { .. })
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

// rq-c3da9ee1
#[derive(Debug, Clone)]
pub struct HarmonicBondBuilder;

impl PotentialBuilder for HarmonicBondBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        // Active only when at least one bond uses a harmonic bond type.
        let has_harmonic = cx
            .bond_list
            .bonds
            .iter()
            .any(|b| is_harmonic(cx.bond_types, b.bond_type_index));
        if !has_harmonic {
            return Ok(None);
        }
        let state = HarmonicBondState::new(cx.gpu, cx.bond_list, cx.bond_types)?;
        Ok(Some(Box::new(state)))
    }
}

/// Harmonic per-bond force fragment for the JIT-composed bonded module.
/// The functor exposes `evaluate(r2, r, bond_type_index, dx, dy, dz,
/// fmag, u_k, w_k)` per the contract in
/// `rqm/forces/jit-composed-intramolecular.md`.
// rq-ca10a975
pub fn harmonic_bonded_force_fragment() -> BondedForceFragment {
    let functor_source = r#"
struct HarmonicPairFunctor {
    const Real *bond_k;
    const Real *bond_r0;

    __device__ inline void evaluate(
        Real r2, Real r,
        unsigned int bond_type_index,
        Real dx, Real dy, Real dz,
        Real &fmag,
        Real &u_k,
        Real &w_k) const
    {
        (void) dx; (void) dy; (void) dz;
        // Defensive guard for a near-degenerate bond length; the force
        // direction is undefined as r -> 0. (The composed outer loop
        // already returns early on r2 == 0.)
        if (r < R(1.0e-7)) {
            fmag = R(0.0);
            u_k  = R(0.0);
            w_k  = R(0.0);
            return;
        }
        Real k  = bond_k[bond_type_index];
        Real r0 = bond_r0[bond_type_index];
        Real dr = r - r0;
        // F_radial = -dU/dr = -k*dr for U = 1/2 k dr^2. `fmag` is the
        // per-component factor produced by dividing by r, so the
        // outer-loop body multiplying by (dx, dy, dz) yields the
        // Cartesian force on atom_i. The 1/2 convention is absorbed
        // analytically here (no half-factor in the force); it survives
        // only as the 0.5 literal in the energy term below.
        fmag = -k * dr / r;
        u_k  = R(0.5) * k * dr * dr;
        w_k  = fmag * r2;
    }
};
"#;
    // `entry_point_args` and `functor_init_source` are generated from
    // `harmonic_arg_schema()`, the same schema `bind_bonded_force_args` is
    // validated against; the functor field names in `functor_source` above
    // must match the schema's `functor_field` entries.
    let schema = harmonic_arg_schema();
    BondedForceFragment {
        label: LABEL,
        functor_struct_name: "HarmonicPairFunctor",
        functor_source: functor_source.to_string(),
        entry_point_args: schema.entry_point_args(),
        functor_init_source: schema.functor_init_source(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The CUDA argument declarations and local-functor initialisation the
    // bonded composer expects for the harmonic slot. The schema-generated
    // output must equal these so the composed bonded module compiles to
    // the expected per-bond kernel.
    const EXPECTED_ENTRY_POINT_ARGS: &str = r#"    const Real *harmonic_bond_k,
    const Real *harmonic_bond_r0,
"#;

    const EXPECTED_FUNCTOR_INIT_SOURCE: &str = r#"    functor.bond_k = harmonic_bond_k;
    functor.bond_r0 = harmonic_bond_r0;
"#;

    #[test]
    fn generated_entry_point_args_match_expected() {
        assert_eq!(
            harmonic_arg_schema().entry_point_args(),
            EXPECTED_ENTRY_POINT_ARGS
        );
    }

    #[test]
    fn generated_functor_init_source_is_local_functor() {
        let init = harmonic_arg_schema().functor_init_source();
        assert_eq!(init, EXPECTED_FUNCTOR_INIT_SOURCE);
        // Intramolecular slots use a local `functor`, never the
        // pair-force composite member.
        assert!(!init.contains("composite."));
    }
}
