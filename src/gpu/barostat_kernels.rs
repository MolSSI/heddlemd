// rq-2093594f
//
// Kernel handles for the `barostat` PTX module (`kernels/barostat.cu`).
// Two kernels (`virial_sum_reduce`, `rescale_positions`) shared by the
// Berendsen and c-rescale barostats and by the MTK barostat substep.
// Lives in `src/gpu/` (rather than inside one barostat's module)
// because no single consumer is its natural owner.

// rq-0d8c8688
//
// The `barostat` PTX module backs both barostats and the MTK barostat
// substep; the scalar-virial reduction, per-log-row potential-energy
// reduction, µ computation, and barostat position rescale stages are
// owned here.
crate::gpu_kernels! {
    module: "barostat",
    ptx: crate::kernels::BAROSTAT,
    struct: BarostatKernels,
    kernels: [
        virial_sum_reduce,
        virial_sum_reduce_partials,
        rescale_positions,
        rescale_positions_device_factor,
        multiply_lattice_isotropic,
        c_rescale_compute_mu,
        berendsen_compute_mu,
    ],
    stages: {
        VIRIAL_SUM_REDUCE                    = "virial_sum_reduce",
        POTENTIAL_ENERGY_REDUCE              = "potential_energy_reduce",
        C_RESCALE_COMPUTE_MU                 = "c_rescale_compute_mu_and_rescale_lattice",
        BERENDSEN_BAROSTAT_COMPUTE_MU        = "berendsen_compute_mu_and_rescale_lattice",
        BERENDSEN_BAROSTAT_RESCALE_POSITIONS = "berendsen_barostat_rescale_positions",
        C_RESCALE_BAROSTAT_RESCALE_POSITIONS = "c_rescale_barostat_rescale_positions",
    },
}
