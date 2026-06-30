// rq-2093594f
//
// Kernel handles for the `fill` PTX module (`kernels/fill.cu`). The
// smoke-test kernel that validates the full toolchain.

crate::gpu_kernels! {
    module: "fill",
    ptx: crate::kernels::FILL,
    struct: FillKernels,
    kernels: [fill],
    stages: {},
}
