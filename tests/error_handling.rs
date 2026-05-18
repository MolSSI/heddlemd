// rq-fc0de81b — tests for the project-wide error-handling convention.
use std::error::Error;
use std::path::PathBuf;

use dynamics::forces::{TopologyFileError, ForceFieldError, NeighborListError};
use dynamics::gpu::GpuError;
use dynamics::integrator::IntegratorError;
use dynamics::io::{ConfigError, InitStateError, LogWriterError, TrajectoryWriterError};
use dynamics::pbc::SimulationBoxError;
use dynamics::runner::RunnerError;
use dynamics::state::ParticleStateError;
use dynamics::timings::{TimingsError, TimingsWriterError};

#[test] // rq-fdf7a255
fn thiserror_is_a_declared_dependency() {
    let manifest =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
            .expect("read Cargo.toml");
    let parsed: toml::Value = manifest.parse().expect("parse Cargo.toml");
    let deps = parsed
        .get("dependencies")
        .and_then(|d| d.as_table())
        .expect("[dependencies] table");
    assert!(
        deps.contains_key("thiserror"),
        "thiserror must appear under [dependencies]"
    );
}

#[test] // rq-494626a0
fn every_governed_error_type_implements_std_error() {
    fn assert_error<E: std::error::Error + 'static>() {}
    assert_error::<GpuError>();
    assert_error::<ConfigError>();
    assert_error::<InitStateError>();
    assert_error::<TopologyFileError>();
    assert_error::<ParticleStateError>();
    assert_error::<SimulationBoxError>();
    assert_error::<NeighborListError>();
    assert_error::<ForceFieldError>();
    assert_error::<IntegratorError>();
    assert_error::<TimingsError>();
    assert_error::<TimingsWriterError>();
    assert_error::<TrajectoryWriterError>();
    assert_error::<LogWriterError>();
    assert_error::<RunnerError>();
}

#[test] // rq-3298bdc5
fn config_error_invalid_value_renders_as_prose() {
    let e = ConfigError::InvalidValue {
        field: "simulation.dt".to_string(),
        reason: "must be finite and positive".to_string(),
    };
    assert_eq!(
        format!("{e}"),
        "invalid value for `simulation.dt`: must be finite and positive"
    );
}

#[test] // rq-af191d10
fn neighbor_list_error_too_many_cells_renders_as_prose() {
    let e = NeighborListError::TooManyCells {
        n_cells_total: 4_298_942_376,
        max_supported: 4_294_967_295,
    };
    assert_eq!(
        format!("{e}"),
        "cell grid has 4298942376 cells, exceeding the device limit of 4294967295"
    );
}

#[test] // rq-77c04470
fn runner_error_output_exists_renders_as_prose() {
    let e = RunnerError::OutputExists {
        path: PathBuf::from("/tmp/sim/argon.out.xyz"),
    };
    assert_eq!(
        format!("{e}"),
        "output file already exists: `/tmp/sim/argon.out.xyz`"
    );
}

#[test] // rq-5d9d7f83
fn display_output_is_distinct_from_debug_rendering() {
    let config = ConfigError::InvalidValue {
        field: "simulation.dt".to_string(),
        reason: "must be finite and positive".to_string(),
    };
    let cells = NeighborListError::TooManyCells {
        n_cells_total: 4_298_942_376,
        max_supported: 4_294_967_295,
    };
    let output = RunnerError::OutputExists {
        path: PathBuf::from("/tmp/sim/argon.out.xyz"),
    };
    assert_ne!(format!("{config}"), format!("{config:?}"));
    assert_ne!(format!("{cells}"), format!("{cells:?}"));
    assert_ne!(format!("{output}"), format!("{output:?}"));
}

#[test] // rq-5d6085ba
fn from_generates_a_direct_error_conversion() {
    let nle = NeighborListError::TooManyCells {
        n_cells_total: 1,
        max_supported: 0,
    };
    let ffe: ForceFieldError = nle.into();
    assert!(matches!(
        ffe,
        ForceFieldError::NeighborList(NeighborListError::TooManyCells {
            n_cells_total: 1,
            max_supported: 0,
        })
    ));
}

#[test] // rq-8abcd634
fn driver_error_to_timings_error_conversion_is_retained() {
    fn assert_from<T: From<cudarc::driver::DriverError>>() {}
    assert_from::<TimingsError>();
}

#[test] // rq-4f8e37af
fn wrapped_error_is_reachable_through_source() {
    let ffe = ForceFieldError::NeighborList(NeighborListError::TooManyCells {
        n_cells_total: 1,
        max_supported: 0,
    });
    let source = ffe.source().expect("ForceFieldError has a source");
    let inner = source
        .downcast_ref::<NeighborListError>()
        .expect("source downcasts to NeighborListError");
    assert!(matches!(inner, NeighborListError::TooManyCells { .. }));
}

#[test] // rq-7dd509c8
fn source_walks_the_full_cause_chain() {
    let runner = RunnerError::ForceField(ForceFieldError::NeighborList(
        NeighborListError::TooManyCells {
            n_cells_total: 1,
            max_supported: 0,
        },
    ));
    let lvl1 = runner.source().expect("RunnerError has a source");
    assert!(
        lvl1.downcast_ref::<ForceFieldError>().is_some(),
        "first source is the ForceFieldError"
    );
    let lvl2 = lvl1.source().expect("ForceFieldError has a source");
    assert!(
        lvl2.downcast_ref::<NeighborListError>().is_some(),
        "second source is the NeighborListError"
    );
    assert!(
        lvl2.source().is_none(),
        "NeighborListError::TooManyCells terminates the chain"
    );
}

#[test] // rq-244fceb1
fn wrapping_variant_display_delegates_to_inner_error() {
    let runner = RunnerError::ForceField(ForceFieldError::NeighborList(
        NeighborListError::TooManyCells {
            n_cells_total: 4_298_942_376,
            max_supported: 4_294_967_295,
        },
    ));
    assert_eq!(
        format!("{runner}"),
        "cell grid has 4298942376 cells, exceeding the device limit of 4294967295"
    );
}
