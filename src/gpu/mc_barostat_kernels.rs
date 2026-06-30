// Kernel handle for the `mc_barostat` PTX module
// (`kernels/mc_barostat.cu`). Backs the Monte-Carlo barostat's
// rigid molecular-centre-of-mass volume-scale move. See
// `rqm/integration/mc-barostat.md`.
crate::gpu_kernels! {
    module: "mc_barostat",
    ptx: crate::kernels::MC_BAROSTAT,
    struct: McBarostatKernels,
    kernels: [
        mc_barostat_scale_molecule_com,
    ],
    stages: {
        MC_BAROSTAT_SCALE_COM = "mc_barostat_scale_molecule_com",
    },
}
