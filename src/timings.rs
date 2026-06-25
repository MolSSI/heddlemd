// rq-bbb62e9c rq-410afcd3 rq-4f5643f1
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::CudaDevice;
use cudarc::driver::result::event;
use cudarc::driver::sys::{CUevent, CUevent_flags};

use crate::gpu::{GpuContext, GpuError};

// rq-dc8a0ff7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelStage {
    name: &'static str,
}

impl KernelStage {
    pub const fn new(name: &'static str) -> Self {
        KernelStage { name }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const VV_KICK_DRIFT: KernelStage = KernelStage::new("vv_kick_drift");
    pub const VV_KICK: KernelStage = KernelStage::new("vv_kick");
    pub const VV_KICK_DRIFT_LOSSLESS: KernelStage =
        KernelStage::new("vv_kick_drift_lossless");
    pub const VV_KICK_LOSSLESS: KernelStage = KernelStage::new("vv_kick_lossless");
    pub const LJ_PAIR_FORCE: KernelStage = KernelStage::new("lj_pair_force");
    pub const COULOMB_PAIR_FORCE: KernelStage = KernelStage::new("coulomb_pair_force");
    pub const SPME_REAL_PAIR_FORCE: KernelStage =
        KernelStage::new("spme_real_pair_force");
    /// rq-9f309378
    pub const JIT_COMPOSED_PAIR_FORCE: KernelStage =
        KernelStage::new("jit_composed_pair_force");
    /// rq-2d2eaf72
    pub const JIT_COMPOSED_BONDED_FORCE: KernelStage =
        KernelStage::new("jit_composed_bonded_force");
    /// rq-2d2eaf72
    pub const JIT_COMPOSED_ANGLE_FORCE: KernelStage =
        KernelStage::new("jit_composed_angle_force");
    /// rq-8ac9773d — JIT-composed post-force per-particle kernel
    /// (integrator + thermostat + barostat per-particle work fused
    /// into a single launch). See
    /// `rqm/integration/jit-composed-post-force.md`.
    pub const JIT_COMPOSED_POST_FORCE: KernelStage =
        KernelStage::new("jit_composed_post_force");
    pub const SPME_RECIP_PIPELINE: KernelStage =
        KernelStage::new("spme_recip_pipeline");
    pub const SPME_FORCE_GATHER: KernelStage = KernelStage::new("spme_force_gather");
    pub const LANGEVIN_KICK_HALF: KernelStage = KernelStage::new("langevin_kick_half");
    pub const LANGEVIN_DRIFT_HALF: KernelStage = KernelStage::new("langevin_drift_half");
    pub const LANGEVIN_OU_STEP: KernelStage = KernelStage::new("langevin_ou_step");
    pub const REDUCE_BOND_FORCES: KernelStage = KernelStage::new("reduce_bond_forces");
    pub const REDUCE_ANGLE_FORCES: KernelStage =
        KernelStage::new("reduce_angle_forces");
    pub const KINETIC_ENERGY_REDUCE: KernelStage =
        KernelStage::new("kinetic_energy_reduce");
    pub const NHC_RESCALE_VELOCITIES: KernelStage =
        KernelStage::new("nhc_rescale_velocities");
    pub const CSVR_RESCALE_VELOCITIES: KernelStage =
        KernelStage::new("csvr_rescale_velocities");
    pub const ANDERSEN_RESAMPLE: KernelStage =
        KernelStage::new("andersen_resample");
    pub const BERENDSEN_RESCALE_VELOCITIES: KernelStage =
        KernelStage::new("berendsen_rescale_velocities");
    // Thermostat/barostat scalar-prep stages. These wrap the device
    // scalar kernels that each slot's `apply_*` launches (the per-particle
    // rescale is folded into the JIT-composed post-force kernel and timed
    // separately). rq-5f59fa80
    pub const CSVR_SAMPLE_AND_FACTOR: KernelStage =
        KernelStage::new("csvr_sample_and_factor");
    pub const BERENDSEN_COMPUTE_FACTOR: KernelStage =
        KernelStage::new("berendsen_compute_factor");
    pub const C_RESCALE_COMPUTE_MU: KernelStage =
        KernelStage::new("c_rescale_compute_mu_and_rescale_lattice");
    pub const BERENDSEN_BAROSTAT_COMPUTE_MU: KernelStage =
        KernelStage::new("berendsen_compute_mu_and_rescale_lattice");
    // rq-0d8c8688
    pub const VIRIAL_SUM_REDUCE: KernelStage =
        KernelStage::new("virial_sum_reduce");
    pub const POTENTIAL_ENERGY_REDUCE: KernelStage =
        KernelStage::new("potential_energy_reduce");
    pub const BERENDSEN_BAROSTAT_RESCALE_POSITIONS: KernelStage =
        KernelStage::new("berendsen_barostat_rescale_positions");
    // rq-11f5dfd1
    pub const C_RESCALE_BAROSTAT_RESCALE_POSITIONS: KernelStage =
        KernelStage::new("c_rescale_barostat_rescale_positions");
    // rq-3b6d5001
    pub const MTK_NPT_RESCALE_VELOCITIES: KernelStage =
        KernelStage::new("mtk_npt_rescale_velocities");
    pub const MTK_NPT_VELOCITY_HALF_KICK: KernelStage =
        KernelStage::new("mtk_npt_velocity_half_kick");
    pub const MTK_NPT_POSITION_DRIFT: KernelStage =
        KernelStage::new("mtk_npt_position_drift");
    // rq-157e59ad
    pub const SHAKE_SNAPSHOT: KernelStage = KernelStage::new("shake_snapshot");
    pub const SHAKE_POSITIONS: KernelStage = KernelStage::new("shake_positions");
    pub const RATTLE_VELOCITIES: KernelStage = KernelStage::new("rattle_velocities");
    pub const CONSTRAINT_VIRIAL_SCATTER: KernelStage =
        KernelStage::new("constraint_virial_scatter");
    pub const SHAKE_POSITIONS_NO_VELOCITY: KernelStage =
        KernelStage::new("shake_positions_no_velocity");
    pub const SD_F_MAX_REDUCTION: KernelStage = KernelStage::new("sd_f_max_reduction");
    pub const SD_COMPUTE_STEP: KernelStage = KernelStage::new("sd_compute_step");
    pub const SD_SNAPSHOT: KernelStage = KernelStage::new("sd_snapshot");
    pub const SD_RESTORE: KernelStage = KernelStage::new("sd_restore");
    pub const COMBINE_CLASS_TOTALS: KernelStage = KernelStage::new("combine_class_totals");
    pub const CLASS_ACCUMULATOR_MEMSET: KernelStage =
        KernelStage::new("class_accumulator_memset");
    pub const NEIGHBOR_DISPLACEMENT_SQUARED: KernelStage =
        KernelStage::new("neighbor_displacement_check_flag");
    pub const NEIGHBOR_LIST_BUILD: KernelStage = KernelStage::new("neighbor_list_build");
    pub const COPY_POSITIONS_INTO_REFERENCE: KernelStage =
        KernelStage::new("copy_positions_into_reference");
    pub const SCATTER_POSITIONS_TO_TILE_ORDER: KernelStage =
        KernelStage::new("scatter_positions_to_tile_order");
    pub const FINALIZE_PACKED_FORCES: KernelStage =
        KernelStage::new("finalize_packed_forces");

    pub const ORDER: &'static [KernelStage] = &[
        Self::VV_KICK_DRIFT,
        Self::VV_KICK_DRIFT_LOSSLESS,
        Self::LANGEVIN_KICK_HALF,
        Self::LANGEVIN_DRIFT_HALF,
        Self::LANGEVIN_OU_STEP,
        Self::NEIGHBOR_DISPLACEMENT_SQUARED,
        Self::COPY_POSITIONS_INTO_REFERENCE,
        Self::NEIGHBOR_LIST_BUILD,
        Self::CLASS_ACCUMULATOR_MEMSET,
        Self::SCATTER_POSITIONS_TO_TILE_ORDER,
        Self::LJ_PAIR_FORCE,
        Self::COULOMB_PAIR_FORCE,
        Self::SPME_REAL_PAIR_FORCE,
        Self::JIT_COMPOSED_PAIR_FORCE,
        Self::FINALIZE_PACKED_FORCES,
        Self::JIT_COMPOSED_BONDED_FORCE,
        Self::JIT_COMPOSED_ANGLE_FORCE,
        Self::JIT_COMPOSED_POST_FORCE,
        Self::SPME_RECIP_PIPELINE,
        Self::SPME_FORCE_GATHER,
        Self::REDUCE_BOND_FORCES,
        Self::REDUCE_ANGLE_FORCES,
        Self::KINETIC_ENERGY_REDUCE,
        Self::NHC_RESCALE_VELOCITIES,
        Self::CSVR_RESCALE_VELOCITIES,
        Self::ANDERSEN_RESAMPLE,
        Self::BERENDSEN_RESCALE_VELOCITIES,
        Self::CSVR_SAMPLE_AND_FACTOR,
        Self::BERENDSEN_COMPUTE_FACTOR,
        Self::C_RESCALE_COMPUTE_MU,
        Self::BERENDSEN_BAROSTAT_COMPUTE_MU,
        Self::VIRIAL_SUM_REDUCE,
        Self::POTENTIAL_ENERGY_REDUCE,
        Self::BERENDSEN_BAROSTAT_RESCALE_POSITIONS,
        Self::C_RESCALE_BAROSTAT_RESCALE_POSITIONS,
        Self::MTK_NPT_RESCALE_VELOCITIES,
        Self::MTK_NPT_VELOCITY_HALF_KICK,
        Self::MTK_NPT_POSITION_DRIFT,
        Self::SHAKE_SNAPSHOT,
        Self::SHAKE_POSITIONS,
        Self::RATTLE_VELOCITIES,
        Self::CONSTRAINT_VIRIAL_SCATTER,
        Self::SHAKE_POSITIONS_NO_VELOCITY,
        Self::SD_F_MAX_REDUCTION,
        Self::SD_COMPUTE_STEP,
        Self::SD_SNAPSHOT,
        Self::SD_RESTORE,
        Self::COMBINE_CLASS_TOTALS,
        Self::VV_KICK,
        Self::VV_KICK_LOSSLESS,
    ];
}

// rq-d29f2811
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostStage {
    name: &'static str,
}

impl HostStage {
    pub const fn new(name: &'static str) -> Self {
        HostStage { name }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const CONFIG_LOAD: HostStage = HostStage::new("config_load");
    pub const INIT_LOAD: HostStage = HostStage::new("init_load");
    pub const GPU_INIT: HostStage = HostStage::new("gpu_init");
    pub const VELOCITY_GENERATION: HostStage = HostStage::new("velocity_generation");
    pub const HOST_TO_DEVICE_UPLOAD: HostStage = HostStage::new("host_to_device_upload");
    pub const DEVICE_TO_HOST_DOWNLOAD: HostStage =
        HostStage::new("device_to_host_download");
    pub const TRAJECTORY_WRITE: HostStage = HostStage::new("trajectory_write");
    pub const LOG_WRITE: HostStage = HostStage::new("log_write");
    pub const NEIGHBOR_LIST_REBUILD: HostStage = HostStage::new("neighbor_list_rebuild");
    pub const TOTAL_RUNTIME: HostStage = HostStage::new("total_runtime");

    pub const ORDER: &'static [HostStage] = &[
        Self::HOST_TO_DEVICE_UPLOAD,
        Self::DEVICE_TO_HOST_DOWNLOAD,
        Self::NEIGHBOR_LIST_REBUILD,
        Self::TRAJECTORY_WRITE,
        Self::LOG_WRITE,
        Self::VELOCITY_GENERATION,
        Self::CONFIG_LOAD,
        Self::INIT_LOAD,
        Self::GPU_INIT,
        Self::TOTAL_RUNTIME,
    ];
}

// rq-0dab90aa
#[derive(Debug, Clone)]
pub struct StageStats {
    pub name: String,
    pub count: u64,
    pub total_ns: u128,
    pub min_ns: u64,
    pub max_ns: u64,
}

impl StageStats {
    pub fn mean_us(&self) -> f64 {
        debug_assert!(self.count > 0);
        (self.total_ns as f64) / 1000.0 / (self.count as f64)
    }

    fn total_ms(&self) -> f64 {
        (self.total_ns as f64) / 1_000_000.0
    }
}

// rq-7453115b
#[derive(Debug, Clone)]
pub struct TimingsReport {
    pub stages: Vec<StageStats>,
}

// rq-779092ca rq-e1ceb5c0 rq-6cf916af
#[derive(Debug, thiserror::Error)]
pub enum TimingsError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
}

// Converting a raw `DriverError` into a `TimingsError` routes through
// `GpuError`; that two-hop conversion cannot be expressed with `#[from]`
// and stays a hand-written impl alongside the derived `From<GpuError>`.
impl From<cudarc::driver::DriverError> for TimingsError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        TimingsError::Gpu(GpuError(e))
    }
}

// rq-ec06c8e1 rq-e1ceb5c0
#[derive(Debug, thiserror::Error)]
pub enum TimingsWriterError {
    #[error("output file already exists: `{}`", .path.display())]
    OutputExists { path: PathBuf },
    #[error("failed to write timings file: {0}")]
    Io(String),
}

#[derive(Debug, Clone, Copy, Default)]
struct Accumulator {
    count: u64,
    total_ns: u128,
    min_ns: u64,
    max_ns: u64,
}

impl Accumulator {
    fn record_ns(&mut self, ns: u64) {
        if self.count == 0 {
            self.min_ns = ns;
            self.max_ns = ns;
        } else {
            if ns < self.min_ns {
                self.min_ns = ns;
            }
            if ns > self.max_ns {
                self.max_ns = ns;
            }
        }
        self.count += 1;
        self.total_ns = self.total_ns.saturating_add(ns as u128);
    }
}

#[derive(Debug)]
struct KernelStageState {
    start: CUevent,
    stop: CUevent,
    outstanding_stop: bool,
    acc: Accumulator,
    /// Number of `kernel_stop` calls observed during the active
    /// capture iteration. `Timings::record_graph_replays` multiplies
    /// this by the replay count to advance `acc.count` without
    /// re-firing any events.
    captured_stops_per_replay: u32,
}

// rq-baf03449
pub struct Timings {
    device: Arc<CudaDevice>,
    kernel_states: HashMap<KernelStage, KernelStageState>,
    host_acc: HashMap<HostStage, Accumulator>,
    /// When `true`, the runner is currently in the dry capture
    /// iteration of a graph-eligible MD phase. `kernel_start` skips
    /// the prior-stop synchronise, and `kernel_stop` increments each
    /// stage's `captured_stops_per_replay` instead of setting
    /// `outstanding_stop`. See `cuda-graphs.md`.
    in_capture: bool,
}

impl std::fmt::Debug for Timings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timings")
            .field("kernel_states", &self.kernel_states)
            .field("host_acc", &self.host_acc)
            .finish_non_exhaustive()
    }
}

impl Timings {
    // rq-8a9c44f8
    pub fn new(gpu: &GpuContext) -> Result<Self, TimingsError> {
        let device = gpu.device.clone();
        let mut kernel_states: HashMap<KernelStage, KernelStageState> =
            HashMap::with_capacity(KernelStage::ORDER.len());
        for &stage in KernelStage::ORDER {
            let start = event::create(CUevent_flags::CU_EVENT_DEFAULT)?;
            let stop = event::create(CUevent_flags::CU_EVENT_DEFAULT)?;
            kernel_states.insert(
                stage,
                KernelStageState {
                    start,
                    stop,
                    outstanding_stop: false,
                    acc: Accumulator::default(),
                    captured_stops_per_replay: 0,
                },
            );
        }
        let mut host_acc: HashMap<HostStage, Accumulator> =
            HashMap::with_capacity(HostStage::ORDER.len());
        for &stage in HostStage::ORDER {
            host_acc.insert(stage, Accumulator::default());
        }
        Ok(Timings {
            device,
            kernel_states,
            host_acc,
            in_capture: false,
        })
    }

    /// Enter capture mode. `kernel_start` skips the prior-stop
    /// synchronise; `kernel_stop` accumulates per-stage replay counts
    /// instead of marking `outstanding_stop`. Per-replay counts reset
    /// at the start of every capture.
    pub fn begin_capture(&mut self) {
        self.in_capture = true;
        for state in self.kernel_states.values_mut() {
            state.captured_stops_per_replay = 0;
        }
    }

    /// Leave capture mode. The per-stage capture counts remain set
    /// until the next `begin_capture` and are consumed by
    /// `record_graph_replays`.
    pub fn end_capture(&mut self) {
        self.in_capture = false;
    }

    /// Bumps each `KernelStage`'s accumulator sample count by
    /// `captured_stops_per_replay × n_launches`, recording zero
    /// durations. Used by the runner's batched graph-replay loop to
    /// keep the `.timings` file's sample counts consistent with the
    /// per-step launch path; per-kernel elapsed durations are not
    /// collected under graph mode (see `cuda-graphs.md`).
    pub fn record_graph_replays(&mut self, n_launches: u32) {
        for state in self.kernel_states.values_mut() {
            let total = state.captured_stops_per_replay as u64 * n_launches as u64;
            for _ in 0..total {
                state.acc.record_ns(0);
            }
        }
    }

    // rq-58981e16
    pub fn kernel_start(&mut self, stage: KernelStage) -> Result<(), TimingsError> {
        let in_capture = self.in_capture;
        let state = self
            .kernel_states
            .get_mut(&stage)
            .unwrap_or_else(|| panic!("unknown KernelStage: {:?}", stage.name()));
        let stream = *self.device.cu_stream();
        if !in_capture && state.outstanding_stop {
            // Resolving the prior sample requires synchronising on
            // `stop`. When the runner is capturing this is forbidden;
            // we skip the resolve path entirely (see `in_capture`).
            let start = state.start;
            let stop = state.stop;
            unsafe { event::synchronize(stop)? };
            let elapsed_ms = unsafe { event::elapsed(start, stop)? };
            let ns = if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
                (elapsed_ms as f64 * 1_000_000.0).round() as u64
            } else {
                0
            };
            state.acc.record_ns(ns);
            state.outstanding_stop = false;
        }
        unsafe { event::record(state.start, stream)? };
        Ok(())
    }

    // rq-b17e6de6
    pub fn kernel_stop(&mut self, stage: KernelStage) -> Result<(), TimingsError> {
        let in_capture = self.in_capture;
        let state = self
            .kernel_states
            .get_mut(&stage)
            .unwrap_or_else(|| panic!("unknown KernelStage: {:?}", stage.name()));
        let stream = *self.device.cu_stream();
        unsafe { event::record(state.stop, stream)? };
        if in_capture {
            state.captured_stops_per_replay += 1;
        } else {
            state.outstanding_stop = true;
        }
        Ok(())
    }

    // rq-037a9326
    pub fn record_host(&mut self, stage: HostStage, duration: Duration) {
        let ns = duration.as_nanos();
        // Clamp to u64 — single-stage measurements above ~584 years would
        // overflow, which is well outside any plausible run.
        let ns_u64 = ns.min(u64::MAX as u128) as u64;
        let acc = self
            .host_acc
            .get_mut(&stage)
            .unwrap_or_else(|| panic!("unknown HostStage: {:?}", stage.name()));
        acc.record_ns(ns_u64);
    }

    /// Marks every stage's outstanding stop event as already drained
    /// without synchronising. Used by the CUDA graph capture path to
    /// invalidate event-record-node references left over from the
    /// captured iteration; those events fired inside graph execution
    /// and cannot be measured by the host's `event::elapsed` after
    /// the graph instance is destroyed.
    pub fn forget_outstanding(&mut self) {
        for stage in KernelStage::ORDER {
            let state = self.kernel_states.get_mut(stage).expect("stage present");
            state.outstanding_stop = false;
        }
    }

    /// Drains every stage's outstanding start/stop event pair into its
    /// accumulator. Used by the CUDA graph capture path to flush all
    /// outstanding events before `cuStreamBeginCapture_v2` — the per-
    /// stage `event::synchronize` call inside `kernel_start` is not
    /// permitted inside a captured region, so we settle everything
    /// before capture begins.
    pub fn drain_outstanding(&mut self) -> Result<(), TimingsError> {
        for stage in KernelStage::ORDER {
            let state = self.kernel_states.get_mut(stage).expect("stage present");
            if state.outstanding_stop {
                let start = state.start;
                let stop = state.stop;
                unsafe { event::synchronize(stop)? };
                let elapsed_ms = unsafe { event::elapsed(start, stop)? };
                let ns = if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
                    (elapsed_ms as f64 * 1_000_000.0).round() as u64
                } else {
                    0
                };
                state.acc.record_ns(ns);
                state.outstanding_stop = false;
            }
        }
        Ok(())
    }

    // rq-c4845f90
    pub fn finalize(&mut self) -> Result<TimingsReport, TimingsError> {
        for stage in KernelStage::ORDER {
            let state = self.kernel_states.get_mut(stage).expect("stage present");
            if state.outstanding_stop {
                let start = state.start;
                let stop = state.stop;
                unsafe { event::synchronize(stop)? };
                let elapsed_ms = unsafe { event::elapsed(start, stop)? };
                let ns = if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
                    (elapsed_ms as f64 * 1_000_000.0).round() as u64
                } else {
                    0
                };
                state.acc.record_ns(ns);
                state.outstanding_stop = false;
            }
        }

        let mut stages: Vec<StageStats> = Vec::new();
        for &k in KernelStage::ORDER {
            let acc = self.kernel_states.get(&k).expect("stage present").acc;
            if acc.count > 0 {
                stages.push(StageStats {
                    name: k.name().to_string(),
                    count: acc.count,
                    total_ns: acc.total_ns,
                    min_ns: acc.min_ns,
                    max_ns: acc.max_ns,
                });
            }
        }
        for &h in HostStage::ORDER {
            let acc = *self.host_acc.get(&h).expect("stage present");
            if acc.count > 0 {
                stages.push(StageStats {
                    name: h.name().to_string(),
                    count: acc.count,
                    total_ns: acc.total_ns,
                    min_ns: acc.min_ns,
                    max_ns: acc.max_ns,
                });
            }
        }
        Ok(TimingsReport { stages })
    }
}

impl Drop for Timings {
    fn drop(&mut self) {
        for state in self.kernel_states.values() {
            unsafe {
                let _ = event::destroy(state.start);
                let _ = event::destroy(state.stop);
            }
        }
    }
}

fn ns_to_us(ns: u64) -> f64 {
    (ns as f64) / 1000.0
}

// rq-9b85fa6c
pub fn write_timings_file(
    path: &Path,
    report: &TimingsReport,
) -> Result<(), TimingsWriterError> {
    let file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(TimingsWriterError::OutputExists {
                path: path.to_path_buf(),
            });
        }
        Err(e) => return Err(TimingsWriterError::Io(format!("{}: {}", path.display(), e))),
    };
    let mut w = BufWriter::new(file);
    // rq-56364532
    writeln!(
        w,
        "{:<28} {:>10} {:>14} {:>13} {:>11} {:>11}",
        "stage", "count", "total_ms", "mean_us", "min_us", "max_us"
    )
    .map_err(io_err)?;
    for stats in &report.stages {
        writeln!(
            w,
            "{:<28} {:>10} {:>14.3} {:>13.1} {:>11.1} {:>11.1}",
            stats.name,
            stats.count,
            stats.total_ms(),
            stats.mean_us(),
            ns_to_us(stats.min_ns),
            ns_to_us(stats.max_ns),
        )
        .map_err(io_err)?;
    }
    w.flush().map_err(io_err)?;
    Ok(())
}

fn io_err(e: std::io::Error) -> TimingsWriterError {
    TimingsWriterError::Io(format!("{e}"))
}
