// rq-2093594f
//
// Declarative consolidation of per-subsystem CUDA kernel wiring. Each
// subsystem owns its kernel handles, its loader, its `KernelStage`
// consts, and its `STAGES` slice through one `gpu_kernels!` invocation
// in the subsystem's own file. The central `Kernels` aggregate, its
// `load`, and the `KernelStage::ORDER` registry are expanded from one
// `define_kernels!` manifest in `device.rs`, so the three can never
// drift apart. See `rqm/build-pipeline.md` and
// `rqm/performance-analysis.md`.

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::gpu::GpuError;
use crate::timings::KernelStage;

/// Contract every per-subsystem kernel sub-struct satisfies (implemented
/// by `gpu_kernels!`). `define_kernels!` is generic over it: it composes
/// `Kernels::load` from each field's `load` and `KernelStage::ORDER` from
/// each field's `STAGES`.
// rq-2093594f
pub trait SubsystemKernels: Sized + Clone + core::fmt::Debug {
    /// PTX module name (the `.cu` stem).
    const MODULE: &'static str;
    /// The subsystem's timed stages, in launch order. Empty for a
    /// subsystem that records none.
    const STAGES: &'static [KernelStage];
    /// Load the subsystem's PTX module and pull its function handles.
    fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError>;
}

/// Concatenate `groups` (each subsystem's `STAGES`) into one fixed array
/// of length `N`, in the given order. `N` must equal the summed lengths
/// of `groups`; `define_kernels!` computes it from the manifest. Used to
/// assemble `KernelStage::ORDER` at const-eval time.
// rq-2093594f
pub const fn concat_kernel_stages<const N: usize>(
    groups: &[&[KernelStage]],
) -> [KernelStage; N] {
    let mut out = [KernelStage::new(""); N];
    let mut oi = 0;
    let mut gi = 0;
    while gi < groups.len() {
        let g = groups[gi];
        let mut i = 0;
        while i < g.len() {
            out[oi] = g[i];
            oi += 1;
            i += 1;
        }
        gi += 1;
    }
    out
}

/// Expand one subsystem's kernel-name list and stage list into the
/// sub-struct, its loader, the `KernelStage` consts it owns, the
/// `STAGES` slice, and the `SubsystemKernels` impl. Invoked once per
/// subsystem in the subsystem's home file. See `rqm/build-pipeline.md`.
// rq-2093594f
#[macro_export]
macro_rules! gpu_kernels {
    (
        module: $module:literal,
        ptx: $ptx:expr,
        struct: $Struct:ident,
        kernels: [ $( $(#[$kattr:meta])* $kernel:ident ),* $(,)? ],
        stages: { $( $stage:ident = $stage_name:literal ),* $(,)? } $(,)?
    ) => {
        #[derive(Debug, Clone)]
        pub struct $Struct {
            $( $(#[$kattr])* pub $kernel: ::cudarc::driver::CudaFunction, )*
        }

        impl $crate::timings::KernelStage {
            $(
                pub const $stage: $crate::timings::KernelStage =
                    $crate::timings::KernelStage::new($stage_name);
            )*
        }

        impl $crate::gpu::SubsystemKernels for $Struct {
            const MODULE: &'static str = $module;
            const STAGES: &'static [$crate::timings::KernelStage] = &[
                $( $crate::timings::KernelStage::$stage, )*
            ];

            fn load(
                device: &::std::sync::Arc<::cudarc::driver::CudaDevice>,
            ) -> ::std::result::Result<Self, $crate::gpu::GpuError> {
                let mut names: ::std::vec::Vec<&'static str> = ::std::vec::Vec::new();
                $( $(#[$kattr])* names.push(::core::stringify!($kernel)); )*
                device.load_ptx(
                    ::cudarc::nvrtc::Ptx::from_src($ptx),
                    $module,
                    names.as_slice(),
                )?;
                ::std::result::Result::Ok($Struct {
                    $(
                        $(#[$kattr])*
                        $kernel: $crate::gpu::device::get_func(
                            device,
                            $module,
                            ::core::stringify!($kernel),
                        )?,
                    )*
                })
            }
        }
    };
}

/// Expand the central subsystem manifest into the `Kernels` aggregate,
/// `Kernels::load`, and `KernelStage::ORDER`. Invoked once, in
/// `device.rs`. `KernelStage::ORDER` is the manifest-order concatenation
/// of every subsystem's `STAGES`. See `rqm/build-pipeline.md`.
// rq-2093594f
#[macro_export]
macro_rules! define_kernels {
    ( $( $field:ident : $ty:ty ),* $(,)? ) => {
        #[derive(Debug, Clone)]
        pub struct Kernels {
            $( pub $field: $ty, )*
        }

        impl Kernels {
            // Composes every subsystem's `load` in manifest order; the
            // first failing subsystem short-circuits the rest.
            pub fn load(
                device: &::std::sync::Arc<::cudarc::driver::CudaDevice>,
            ) -> ::std::result::Result<Self, $crate::gpu::GpuError> {
                ::std::result::Result::Ok(Kernels {
                    $(
                        $field: <$ty as $crate::gpu::SubsystemKernels>::load(device)?,
                    )*
                })
            }
        }

        impl $crate::timings::KernelStage {
            pub const ORDER: &'static [$crate::timings::KernelStage] = {
                const COUNT: usize = 0
                    $( + <$ty as $crate::gpu::SubsystemKernels>::STAGES.len() )*;
                const ORDER_ARR: [$crate::timings::KernelStage; COUNT] =
                    $crate::gpu::concat_kernel_stages::<COUNT>(&[
                        $( <$ty as $crate::gpu::SubsystemKernels>::STAGES, )*
                    ]);
                &ORDER_ARR
            };
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::gpu::SubsystemKernels;
    use crate::timings::KernelStage;
    use std::collections::HashSet;

    // The instrumented-subsystem manifest, mirroring `define_kernels!`
    // in `device.rs`: (module-name, STAGES) in manifest order.
    fn manifest() -> Vec<(&'static str, &'static [KernelStage])> {
        vec![
            (
                <crate::gpu::fill::FillKernels as SubsystemKernels>::MODULE,
                <crate::gpu::fill::FillKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::integrator::velocity_verlet::IntegrateKernels as SubsystemKernels>::MODULE,
                <crate::integrator::velocity_verlet::IntegrateKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::forces::spme::SpmeRecipKernels as SubsystemKernels>::MODULE,
                <crate::forces::spme::SpmeRecipKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::integrator::langevin_baoab::LangevinKernels as SubsystemKernels>::MODULE,
                <crate::integrator::langevin_baoab::LangevinKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::forces::morse::MorseKernels as SubsystemKernels>::MODULE,
                <crate::forces::morse::MorseKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::forces::angle::AngleKernels as SubsystemKernels>::MODULE,
                <crate::forces::angle::AngleKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::integrator::nose_hoover_chain::NoseHooverKernels as SubsystemKernels>::MODULE,
                <crate::integrator::nose_hoover_chain::NoseHooverKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::integrator::andersen::AndersenKernels as SubsystemKernels>::MODULE,
                <crate::integrator::andersen::AndersenKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::gpu::barostat_kernels::BarostatKernels as SubsystemKernels>::MODULE,
                <crate::gpu::barostat_kernels::BarostatKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::integrator::mtk_npt::MtkKernels as SubsystemKernels>::MODULE,
                <crate::integrator::mtk_npt::MtkKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::integrator::shake::ShakeKernels as SubsystemKernels>::MODULE,
                <crate::integrator::shake::ShakeKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::integrator::settle::SettleKernels as SubsystemKernels>::MODULE,
                <crate::integrator::settle::SettleKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::forces::ForcesKernels as SubsystemKernels>::MODULE,
                <crate::forces::ForcesKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::forces::neighbor_list::NeighborKernels as SubsystemKernels>::MODULE,
                <crate::forces::neighbor_list::NeighborKernels as SubsystemKernels>::STAGES,
            ),
            (
                <crate::minimizer::MinimizeKernels as SubsystemKernels>::MODULE,
                <crate::minimizer::MinimizeKernels as SubsystemKernels>::STAGES,
            ),
        ]
    }

    // rq-73a85df1
    #[test]
    fn subsystem_stages_match_declared_stages() {
        use crate::integrator::settle::SettleKernels;
        assert_eq!(<SettleKernels as SubsystemKernels>::MODULE, "settle");
        assert_eq!(
            <SettleKernels as SubsystemKernels>::STAGES,
            &[
                KernelStage::SETTLE_SNAPSHOT,
                KernelStage::SETTLE_POSITIONS,
                KernelStage::SETTLE_VELOCITIES,
                KernelStage::SETTLE_VIRIAL_SCATTER,
                KernelStage::SETTLE_POSITIONS_NO_VELOCITY,
            ]
        );
    }

    // rq-0919ff0a
    #[test]
    fn subsystem_with_no_stages_contributes_empty_stages() {
        use crate::gpu::fill::FillKernels;
        assert!(<FillKernels as SubsystemKernels>::STAGES.is_empty());
        // It therefore contributes no rows to ORDER: no "fill" kernel
        // names a stage, so none of ORDER's entries originate here.
        assert!(<FillKernels as SubsystemKernels>::STAGES.is_empty());
    }

    // rq-a2b911fc
    #[test]
    fn order_is_manifest_order_concatenation_of_subsystem_stages() {
        let mut expected: Vec<KernelStage> = Vec::new();
        for (_module, stages) in manifest() {
            expected.extend_from_slice(stages);
        }
        assert_eq!(KernelStage::ORDER, expected.as_slice());
    }

    // rq-4a584e03
    #[test]
    fn each_subsystem_stages_is_a_contiguous_run_within_order() {
        let order = KernelStage::ORDER;
        for (module, stages) in manifest() {
            if stages.is_empty() {
                continue;
            }
            // Find the first stage's index in ORDER, then assert the
            // whole STAGES slice appears contiguously from there.
            let start = order
                .iter()
                .position(|s| *s == stages[0])
                .unwrap_or_else(|| panic!("{module}: first stage not found in ORDER"));
            assert_eq!(
                &order[start..start + stages.len()],
                stages,
                "{module}: STAGES is not a contiguous run within ORDER"
            );
        }
    }

    // rq-42ee692a
    #[test]
    fn order_has_no_duplicate_stage() {
        let order = KernelStage::ORDER;
        let mut seen: HashSet<&'static str> = HashSet::new();
        for stage in order {
            assert!(
                seen.insert(stage.name()),
                "duplicate stage in ORDER: {}",
                stage.name()
            );
        }
        // Every stage any subsystem declares is present in ORDER.
        for (_module, stages) in manifest() {
            for stage in stages {
                assert!(
                    order.contains(stage),
                    "stage {} declared by a subsystem is absent from ORDER",
                    stage.name()
                );
            }
        }
    }
}
