use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, reduce_bond_forces,
};
use crate::kernels;
use crate::io::config::BondTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::BondList;
use super::{
    AggregateLevel, BondedForceFragment, BondedScratchView, ForceFieldError, ForceLaunchBuilder,
    ForceLaunchContext, Potential, PotentialBuildContext, PotentialBuilder, SlotOutputView,
};
use crate::precision::Real;

// rq-2361f2b8 rq-ec18d174
#[derive(Debug)]
pub struct MorseBondedState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub bonds: CudaSlice<u32>,
    pub atom_bond_offsets: CudaSlice<u32>,
    pub atom_bond_indices: CudaSlice<u32>,
    pub bond_de: CudaSlice<Real>,
    pub bond_a: CudaSlice<Real>,
    pub bond_re: CudaSlice<Real>,
    pub bond_pair_x: CudaSlice<Real>,
    pub bond_pair_y: CudaSlice<Real>,
    pub bond_pair_z: CudaSlice<Real>,
    pub bond_pair_energy: CudaSlice<Real>,
    pub bond_pair_virial: CudaSlice<Real>,
    pub bond_count: usize,
    pub particle_count: usize,
}

impl MorseBondedState {
    pub fn new(
        gpu: &GpuContext,
        bond_list: &BondList,
        bond_types: &[BondTypeConfig],
    ) -> Result<Self, GpuError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let bond_count = bond_list.bonds.len();
        let particle_count = bond_list.particle_count;

        let mut bonds_flat: Vec<u32> = Vec::with_capacity(3 * bond_count);
        for b in &bond_list.bonds {
            bonds_flat.push(b.atom_i);
            bonds_flat.push(b.atom_j);
            bonds_flat.push(b.bond_type_index);
        }

        let mut de_vec: Vec<Real> = Vec::with_capacity(bond_types.len());
        let mut a_vec: Vec<Real> = Vec::with_capacity(bond_types.len());
        let mut re_vec: Vec<Real> = Vec::with_capacity(bond_types.len());
        for bt in bond_types {
            match bt {
                BondTypeConfig::Morse { de, a, re, .. } => {
                    de_vec.push(*de as Real);
                    a_vec.push(*a as Real);
                    re_vec.push(*re as Real);
                }
            }
        }

        let bonds = htod_or_empty_u32(&device, &bonds_flat)?;
        let atom_bond_offsets = htod_or_empty_u32(&device, &bond_list.atom_bond_offsets)?;
        let atom_bond_indices = htod_or_empty_u32(&device, &bond_list.atom_bond_indices)?;
        let bond_de = htod_or_empty(&device, &de_vec)?;
        let bond_a = htod_or_empty(&device, &a_vec)?;
        let bond_re = htod_or_empty(&device, &re_vec)?;

        let bond_pair_len = 2 * bond_count;
        let bond_pair_x = device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_y = device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_z = device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_energy =
            device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;
        let bond_pair_virial =
            device.alloc_zeros::<Real>(bond_pair_len).map_err(GpuError::from)?;

        Ok(MorseBondedState {
            device,
            kernels,
            bonds,
            atom_bond_offsets,
            atom_bond_indices,
            bond_de,
            bond_a,
            bond_re,
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

impl Potential for MorseBondedState {
    fn label(&self) -> &'static str {
        "morse_bonded"
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
        // JIT-composed bonded module dispatch *before* this method;
        // by the time we get here, the slot's bond-pair scratch
        // buffer holds the per-bond contributions. Only the per-atom
        // reduction is the slot's responsibility.
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

    fn bind_bonded_force_args(
        &self,
        _ctx: &ForceLaunchContext<'_>,
        builder: &mut ForceLaunchBuilder,
    ) {
        // Order MUST match the Morse fragment's entry_point_args.
        builder.push_device_buffer(&self.bond_de);
        builder.push_device_buffer(&self.bond_a);
        builder.push_device_buffer(&self.bond_re);
    }

    fn bonded_scratch(&self) -> Option<BondedScratchView<'_>> {
        Some(BondedScratchView {
            bonds: &self.bonds,
            bond_pair_x: &self.bond_pair_x,
            bond_pair_y: &self.bond_pair_y,
            bond_pair_z: &self.bond_pair_z,
            bond_pair_energy: &self.bond_pair_energy,
            bond_pair_virial: &self.bond_pair_virial,
            bond_count: self.bond_count,
        })
    }
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
pub struct MorseBondedBuilder;

impl PotentialBuilder for MorseBondedBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        if cx.bond_list.is_empty() {
            return Ok(None);
        }
        let state = MorseBondedState::new(cx.gpu, cx.bond_list, cx.bond_types)?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }

    fn bonded_force_fragment(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<BondedForceFragment>, ForceFieldError> {
        if cx.bond_list.is_empty() {
            return Ok(None);
        }
        Ok(Some(morse_bonded_force_fragment()))
    }
}

/// Morse per-bond force fragment for the JIT-composed bonded module.
/// The functor exposes `evaluate(r2, r, bond_type_index, dx, dy, dz,
/// fmag, u_k, w_k)` per the contract in
/// `rqm/forces/jit-composed-intramolecular.md`.
pub fn morse_bonded_force_fragment() -> BondedForceFragment {
    let functor_source = r#"
struct MorsePairFunctor {
    const Real *bond_de;
    const Real *bond_a;
    const Real *bond_re;

    __device__ inline void evaluate(
        Real r2, Real r,
        unsigned int bond_type_index,
        Real dx, Real dy, Real dz,
        Real &fmag,
        Real &u_k,
        Real &w_k) const
    {
        (void) dx; (void) dy; (void) dz;
        Real de = bond_de[bond_type_index];
        Real a  = bond_a[bond_type_index];
        Real re = bond_re[bond_type_index];
        Real e  = Real_exp(-a * (r - re));
        Real one_minus_e = R(1.0) - e;
        // F_radial = -dU/dr = -2*De*a*(1 - e)*e. fmag here is the
        // per-component factor produced by dividing by r (so that
        // the outer-loop body multiplying by (dx, dy, dz) gives the
        // Cartesian force on atom_i).
        fmag = -R(2.0) * de * a * one_minus_e * e / r;
        u_k  = de * one_minus_e * one_minus_e;
        w_k  = fmag * r2;
    }
};
"#;
    let entry_point_args = r#"    const Real *morse_bond_de,
    const Real *morse_bond_a,
    const Real *morse_bond_re,
"#;
    let functor_init_source = r#"    functor.bond_de = morse_bond_de;
    functor.bond_a  = morse_bond_a;
    functor.bond_re = morse_bond_re;
"#;
    BondedForceFragment {
        label: "morse_bonded",
        functor_struct_name: "MorsePairFunctor",
        functor_source: functor_source.to_string(),
        entry_point_args: entry_point_args.to_string(),
        functor_init_source: functor_init_source.to_string(),
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct MorseKernels {
    pub reduce_bond_forces: CudaFunction,
}

impl MorseKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::MORSE),
            "morse",
            &["reduce_bond_forces"],
        )?;
        Ok(MorseKernels {
            reduce_bond_forces: get_func(device, "morse", "reduce_bond_forces")?,
        })
    }
}
