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

    /// Fold `count` samples that all share the duration `ns` into the
    /// accumulator in one shot. Used by the graph-replay timing path,
    /// where one measured replay stands in for a whole batch of identical
    /// replays: the total and count advance by the full batch while
    /// min/max see the single representative value once.
    fn record_ns_bulk(&mut self, ns: u64, count: u64) {
        if count == 0 {
            return;
        }
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
        self.count += count;
        self.total_ns = self
            .total_ns
            .saturating_add((ns as u128).saturating_mul(count as u128));
    }
}

/// Which captured graph a batch of replays came from. The two graphs
/// differ only in the force evaluation's `AggregateLevel`, so a stage
/// that runs only under `ForcesAndScalars` (the potential-energy and
/// virial reductions, the `_fev` pair-force kernel) accrues `.timings`
/// samples only from `ForcesAndScalars` replays. See `cuda-graphs.md`
/// *`Timings` Interaction*.
// rq-9ec19227
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphVariant {
    ForcesOnly,
    ForcesAndScalars,
}

#[derive(Debug)]
struct KernelStageState {
    start: CUevent,
    stop: CUevent,
    outstanding_stop: bool,
    acc: Accumulator,
    /// Number of `kernel_stop` calls observed during the active
    /// capture iteration. Committed to one of the per-variant fields
    /// below by `end_capture`.
    captured_stops_per_replay: u32,
    /// `kernel_stop` count per replay of the forces-only graph,
    /// committed by `end_capture(GraphVariant::ForcesOnly)`.
    captured_stops_forces_only: u32,
    /// `kernel_stop` count per replay of the forces+scalars graph,
    /// committed by `end_capture(GraphVariant::ForcesAndScalars)`.
    captured_stops_forces_and_scalars: u32,
    /// Representative per-replay duration (ns) for this stage under graph
    /// mode, snapshotted from the instrumented calibration steps the
    /// runner executes before the replay loop (see
    /// `snapshot_graph_representatives`). `record_graph_replays` folds
    /// this value in for every replayed step.
    representative_ns: u64,
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
                    captured_stops_forces_only: 0,
                    captured_stops_forces_and_scalars: 0,
                    representative_ns: 0,
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

    /// Leave capture mode, committing the per-stage `kernel_stop`
    /// counts observed during the capture to `variant`'s per-replay
    /// field. Each captured graph (forces-only, forces+scalars) calls
    /// this with its own `variant`; `record_graph_replays` then folds in
    /// the matching counts. See `cuda-graphs.md` *`Timings` Interaction*.
    // rq-9ec19227
    pub fn end_capture(&mut self, variant: GraphVariant) {
        self.in_capture = false;
        for state in self.kernel_states.values_mut() {
            match variant {
                GraphVariant::ForcesOnly => {
                    state.captured_stops_forces_only = state.captured_stops_per_replay;
                }
                GraphVariant::ForcesAndScalars => {
                    state.captured_stops_forces_and_scalars =
                        state.captured_stops_per_replay;
                }
            }
        }
    }

    /// Snapshot each stage's mean accumulated duration as its
    /// representative per-replay duration for the graph-replay loop.
    /// Called once after the runner's instrumented calibration steps
    /// (real per-step launches with live CUDA-event timing) and before
    /// the batched replay loop, so the mean reflects the calibration
    /// samples only. A stage with no calibration sample keeps a zero
    /// representative. See `record_graph_replays` and `cuda-graphs.md`.
    // rq-9ec19227
    pub fn snapshot_graph_representatives(&mut self) {
        for state in self.kernel_states.values_mut() {
            state.representative_ns = if state.acc.count > 0 {
                (state.acc.total_ns / state.acc.count as u128) as u64
            } else {
                0
            };
        }
    }

    /// Folds the calibrated per-kernel timings into a batch of
    /// `n_launches` replays of the `variant` graph. CUDA rejects
    /// `cuEventElapsedTime` on events captured into a graph
    /// (`CUDA_ERROR_INVALID_VALUE`), so the in-graph events cannot be
    /// timed by replaying them. Each stage instead carries a
    /// `representative_ns` measured from the runner's instrumented
    /// calibration steps (see `snapshot_graph_representatives`); this
    /// records that value `captured_stops(variant) × n_launches` times in
    /// one shot. A stage absent from `variant`'s graph has zero captured
    /// stops there and accrues nothing, so a scalar-only stage's
    /// `.timings` sample count tracks the number of `ForcesAndScalars`
    /// replays — the scalar steps — rather than `n_steps`. A stage with no
    /// calibration sample folds in a zero duration. See `cuda-graphs.md`.
    // rq-9ec19227
    pub fn record_graph_replays(&mut self, variant: GraphVariant, n_launches: u32) {
        for state in self.kernel_states.values_mut() {
            let stops = match variant {
                GraphVariant::ForcesOnly => state.captured_stops_forces_only,
                GraphVariant::ForcesAndScalars => {
                    state.captured_stops_forces_and_scalars
                }
            } as u64;
            if stops == 0 {
                continue;
            }
            let count = stops * n_launches as u64;
            state.acc.record_ns_bulk(state.representative_ns, count);
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
