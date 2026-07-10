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

// rq-2093594f
//
// Expand the body of a single-launch kernel wrapper. Invoked inside the
// body of a `pub fn` launch wrapper in `src/gpu/kernels.rs`; the
// wrapper's signature and doc comment are hand-written around it, and
// the macro emits only the launch body (empty guard, handle clone,
// launch configuration, `unsafe` launch, `GpuError` mapping, `Ok(())`).
// Crate-internal: re-exported with `pub(crate) use` below rather than
// `#[macro_export]`, since it is invoked only within `kernels.rs`.
//
// Three grid strategies fix both the empty guard and the launch config:
//
//   grid: per_element(<size>)  — one thread per element. Returns `Ok(())`
//     without launching when `<size> == 0`; else block 256, grid
//     `ceil(size / 256)`. The macro computes the element count and passes
//     it as the FINAL kernel argument, so `args` lists every argument
//     before it. (Computing the count before the argument tuple is also
//     what lets the tuple hold `&mut buffers.*` borrows without a
//     conflicting immutable borrow for the count.)
//
//   grid: single_block(<size>) — one block of `BLOCK_SIZE` threads
//     (grid 1). Empty-guard and trailing-count behaviour as per_element.
//
//   grid: single_thread — grid (1,1,1), block (1,1,1). No empty guard,
//     no trailing count; `args` lists every argument explicitly.
//
// The macro emits no timing calls: kernel timing is bracketed externally
// by the runner and CUDA-graph capture. See `rqm/build-pipeline.md`.
macro_rules! gpu_launch {
    (
        func: $func:expr,
        grid: per_element($size:expr),
        args: ( $( $arg:expr ),* $(,)? ) $(,)?
    ) => {{
        let __n = $size;
        if __n == 0 {
            return ::std::result::Result::Ok(());
        }
        let __n_u32 = __n as u32;
        let __func = ($func).clone();
        let __cfg = $crate::gpu::kernels::launch_config(__n_u32);
        unsafe {
            __func
                .launch(__cfg, ( $( $arg, )* __n_u32 ))
                .map_err($crate::gpu::GpuError::from)?;
        }
        ::std::result::Result::Ok(())
    }};
    (
        func: $func:expr,
        grid: single_block($size:expr),
        args: ( $( $arg:expr ),* $(,)? ) $(,)?
    ) => {{
        let __n = $size;
        if __n == 0 {
            return ::std::result::Result::Ok(());
        }
        let __n_u32 = __n as u32;
        let __func = ($func).clone();
        let __cfg = ::cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: ($crate::gpu::kernels::BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            __func
                .launch(__cfg, ( $( $arg, )* __n_u32 ))
                .map_err($crate::gpu::GpuError::from)?;
        }
        ::std::result::Result::Ok(())
    }};
    (
        func: $func:expr,
        grid: single_thread,
        args: ( $( $arg:expr ),* $(,)? ) $(,)?
    ) => {{
        let __func = ($func).clone();
        let __cfg = ::cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            __func
                .launch(__cfg, ( $( $arg, )* ))
                .map_err($crate::gpu::GpuError::from)?;
        }
        ::std::result::Result::Ok(())
    }};
}
pub(crate) use gpu_launch;

#[cfg(test)]
mod tests {
    use crate::gpu::SubsystemKernels;
    use crate::timings::KernelStage;
    use std::collections::HashSet;

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

    // rq-a2b911fc — `KernelStage::ORDER` is assembled by
    // `concat_kernel_stages` as the in-order concatenation of the
    // subsystem `STAGES` slices (empty slices contribute nothing, and
    // each group's stages stay contiguous). The concatenation logic is
    // verified directly on the const helper, so no hand-mirror of the
    // `define_kernels!` manifest is needed.
    #[test]
    fn concat_kernel_stages_concatenates_groups_in_order() {
        use crate::gpu::concat_kernel_stages;
        const A: KernelStage = KernelStage::new("a");
        const B: KernelStage = KernelStage::new("b");
        const C: KernelStage = KernelStage::new("c");
        // An empty middle group mirrors a stage-less subsystem.
        let out = concat_kernel_stages::<3>(&[&[A], &[], &[B, C]]);
        assert_eq!(out, [A, B, C]);
    }

    // rq-42ee692a
    #[test]
    fn order_has_no_duplicate_stage() {
        let mut seen: HashSet<&'static str> = HashSet::new();
        for stage in KernelStage::ORDER {
            assert!(
                seen.insert(stage.name()),
                "duplicate stage in ORDER: {}",
                stage.name()
            );
        }
    }

    // --- gpu_launch! macro ---
    //
    // The per_element and single_thread grid strategies are exercised
    // end-to-end through the public launch wrappers in
    // `tests/gpu_launch_macro.rs`. The single_block strategy has no
    // production launch wrapper (every device-resident single-block
    // reducer takes a non-count trailing argument and so is hand-written),
    // so it is exercised here directly against the always-loaded `fill`
    // kernel.
    // `gpu_launch!` is in textual macro scope here (defined earlier in
    // this module), so it needs no `use`.
    use crate::gpu::{GpuContext, GpuError, init_device};
    use crate::precision::Real;
    // `gpu_launch!` expands to `__func.launch(...)`; the `LaunchAsync`
    // trait that provides `launch` must be in scope at the call site.
    use cudarc::driver::{CudaSlice, LaunchAsync};

    // A single_block-strategy launch of `fill`: grid (1,1,1), block
    // (BLOCK_SIZE,1,1). The element count is appended by the macro, so
    // `args` lists only the output buffer and the fill value.
    fn fill_single_block(
        ctx: &GpuContext,
        out: &mut CudaSlice<Real>,
        value: Real,
        size: u32,
    ) -> Result<(), GpuError> {
        gpu_launch! {
            func: ctx.kernels.fill.fill,
            grid: single_block(size),
            args: (&mut *out, value),
        }
    }

    // rq-e61e9b5f
    #[test]
    fn single_block_guards_on_empty_and_launches_one_block() {
        let ctx = init_device().expect("init_device");
        let zero = 0.0 as Real;

        // size == 0: the empty guard returns Ok(()) without launching, so
        // the buffer is left untouched.
        let mut buf = ctx.device.alloc_zeros::<Real>(4).expect("alloc");
        fill_single_block(&ctx, &mut buf, 9.0, 0).expect("empty single_block must be Ok(())");
        let mut host = vec![zero; 4];
        ctx.device
            .dtoh_sync_copy_into(&buf, &mut host)
            .expect("download");
        assert_eq!(host, vec![zero; 4], "size 0 must launch nothing");

        // size > BLOCK_SIZE: a single block of BLOCK_SIZE threads runs, so
        // only the first BLOCK_SIZE elements are written even though the
        // count (300) is larger. Elements in [BLOCK_SIZE, 300) staying zero
        // prove the grid is exactly one block, not ceil(300 / BLOCK_SIZE).
        let block = crate::gpu::kernels::BLOCK_SIZE as usize;
        let n = block + 44; // 300: past one block, below two
        let fill_value = 7.0 as Real;
        let mut buf = ctx.device.alloc_zeros::<Real>(n).expect("alloc");
        fill_single_block(&ctx, &mut buf, fill_value, n as u32).expect("single_block launch");
        let mut host = vec![zero; n];
        ctx.device
            .dtoh_sync_copy_into(&buf, &mut host)
            .expect("download");
        for (i, &v) in host.iter().enumerate() {
            if i < block {
                assert_eq!(v, fill_value, "element {i} within the one block must be filled");
            } else {
                assert_eq!(v, zero, "element {i} beyond one block must be untouched");
            }
        }
    }

    // rq-ece77047 — kernel timing is bracketed externally by the runner and
    // CUDA-graph capture, never inside a launch wrapper. The gpu_launch!
    // macro emits no timing calls; this guards against a wrapper in
    // `kernels.rs` reintroducing timing.
    #[test]
    fn launch_wrappers_emit_no_timing_calls() {
        let src = include_str!("kernels.rs");
        for token in ["kernel_start", "kernel_stop", "Timings"] {
            assert!(
                !src.contains(token),
                "src/gpu/kernels.rs must contain no `{token}` call: kernel timing is external"
            );
        }
    }
}
