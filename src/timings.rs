// rq-bbb62e9c rq-410afcd3 rq-4f5643f1
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::CudaDevice;
use cudarc::driver::result::event;
use cudarc::driver::sys::{CUevent, CUevent_flags};

use crate::gpu::GpuError;

// rq-dc8a0ff7
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelStage {
    VvKickDrift,
    VvKick,
    VvKickDriftLossless,
    VvKickLossless,
    LjPairForce,
    ReducePairForces,
    LangevinKickHalf,
    LangevinDriftHalf,
    LangevinOuStep,
}

impl KernelStage {
    fn name(self) -> &'static str {
        match self {
            KernelStage::VvKickDrift => "vv_kick_drift",
            KernelStage::VvKick => "vv_kick",
            KernelStage::VvKickDriftLossless => "vv_kick_drift_lossless",
            KernelStage::VvKickLossless => "vv_kick_lossless",
            KernelStage::LjPairForce => "lj_pair_force",
            KernelStage::ReducePairForces => "reduce_pair_forces",
            KernelStage::LangevinKickHalf => "langevin_kick_half",
            KernelStage::LangevinDriftHalf => "langevin_drift_half",
            KernelStage::LangevinOuStep => "langevin_ou_step",
        }
    }

    fn index(self) -> usize {
        match self {
            KernelStage::VvKickDrift => 0,
            KernelStage::VvKick => 1,
            KernelStage::VvKickDriftLossless => 2,
            KernelStage::VvKickLossless => 3,
            KernelStage::LjPairForce => 4,
            KernelStage::ReducePairForces => 5,
            KernelStage::LangevinKickHalf => 6,
            KernelStage::LangevinDriftHalf => 7,
            KernelStage::LangevinOuStep => 8,
        }
    }
}

// rq-d29f2811
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostStage {
    ConfigLoad,
    InitLoad,
    GpuInit,
    VelocityGeneration,
    HostToDeviceUpload,
    DeviceToHostDownload,
    TrajectoryWrite,
    LogWrite,
    TotalRuntime,
}

impl HostStage {
    fn name(self) -> &'static str {
        match self {
            HostStage::ConfigLoad => "config_load",
            HostStage::InitLoad => "init_load",
            HostStage::GpuInit => "gpu_init",
            HostStage::VelocityGeneration => "velocity_generation",
            HostStage::HostToDeviceUpload => "host_to_device_upload",
            HostStage::DeviceToHostDownload => "device_to_host_download",
            HostStage::TrajectoryWrite => "trajectory_write",
            HostStage::LogWrite => "log_write",
            HostStage::TotalRuntime => "total_runtime",
        }
    }

    fn index(self) -> usize {
        match self {
            HostStage::ConfigLoad => 0,
            HostStage::InitLoad => 1,
            HostStage::GpuInit => 2,
            HostStage::VelocityGeneration => 3,
            HostStage::HostToDeviceUpload => 4,
            HostStage::DeviceToHostDownload => 5,
            HostStage::TrajectoryWrite => 6,
            HostStage::LogWrite => 7,
            HostStage::TotalRuntime => 8,
        }
    }
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

// rq-779092ca
#[derive(Debug)]
pub enum TimingsError {
    Gpu(GpuError),
}

impl std::fmt::Display for TimingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingsError::Gpu(e) => write!(f, "Gpu({e})"),
        }
    }
}

impl std::error::Error for TimingsError {}

impl From<cudarc::driver::DriverError> for TimingsError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        TimingsError::Gpu(GpuError(e))
    }
}

// rq-ec06c8e1
#[derive(Debug)]
pub enum TimingsWriterError {
    OutputExists { path: PathBuf },
    Io(String),
}

impl std::fmt::Display for TimingsWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingsWriterError::OutputExists { path } => {
                write!(f, "OutputExists {{ path: {} }}", path.display())
            }
            TimingsWriterError::Io(s) => write!(f, "Io({s})"),
        }
    }
}

impl std::error::Error for TimingsWriterError {}

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

// rq-baf03449
pub struct Timings {
    device: Arc<CudaDevice>,
    kernel_starts: [CUevent; 9],
    kernel_stops: [CUevent; 9],
    outstanding_stop: [bool; 9],
    kernel_acc: [Accumulator; 9],
    host_acc: [Accumulator; 9],
}

impl std::fmt::Debug for Timings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timings")
            .field("kernel_acc", &self.kernel_acc)
            .field("host_acc", &self.host_acc)
            .finish_non_exhaustive()
    }
}

impl Timings {
    // rq-8a9c44f8
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, TimingsError> {
        let mut kernel_starts = [std::ptr::null_mut(); 9];
        let mut kernel_stops = [std::ptr::null_mut(); 9];
        for i in 0..9 {
            kernel_starts[i] = event::create(CUevent_flags::CU_EVENT_DEFAULT)?;
            kernel_stops[i] = event::create(CUevent_flags::CU_EVENT_DEFAULT)?;
        }
        Ok(Timings {
            device,
            kernel_starts,
            kernel_stops,
            outstanding_stop: [false; 9],
            kernel_acc: [Accumulator::default(); 9],
            host_acc: [Accumulator::default(); 9],
        })
    }

    // rq-58981e16
    pub fn kernel_start(&mut self, stage: KernelStage) -> Result<(), TimingsError> {
        let idx = stage.index();
        if self.outstanding_stop[idx] {
            self.drain_pair(idx)?;
        }
        let stream = *self.device.cu_stream();
        unsafe { event::record(self.kernel_starts[idx], stream)? };
        Ok(())
    }

    // rq-b17e6de6
    pub fn kernel_stop(&mut self, stage: KernelStage) -> Result<(), TimingsError> {
        let idx = stage.index();
        let stream = *self.device.cu_stream();
        unsafe { event::record(self.kernel_stops[idx], stream)? };
        self.outstanding_stop[idx] = true;
        Ok(())
    }

    fn drain_pair(&mut self, idx: usize) -> Result<(), TimingsError> {
        // Synchronize on the stop event so cuEventElapsedTime won't return
        // CUDA_ERROR_NOT_READY.
        unsafe { event::synchronize(self.kernel_stops[idx])? };
        let elapsed_ms =
            unsafe { event::elapsed(self.kernel_starts[idx], self.kernel_stops[idx])? };
        // Convert ms to ns, clamp at zero in case the driver reports a tiny
        // negative.
        let ns = if elapsed_ms.is_finite() && elapsed_ms > 0.0 {
            (elapsed_ms as f64 * 1_000_000.0).round() as u64
        } else {
            0
        };
        self.kernel_acc[idx].record_ns(ns);
        self.outstanding_stop[idx] = false;
        Ok(())
    }

    // rq-037a9326
    pub fn record_host(&mut self, stage: HostStage, duration: Duration) {
        let ns = duration.as_nanos();
        // Clamp to u64 — single-stage measurements above ~584 years would
        // overflow, which is well outside any plausible run.
        let ns_u64 = ns.min(u64::MAX as u128) as u64;
        self.host_acc[stage.index()].record_ns(ns_u64);
    }

    // rq-c4845f90
    pub fn finalize(&mut self) -> Result<TimingsReport, TimingsError> {
        for idx in 0..9 {
            if self.outstanding_stop[idx] {
                self.drain_pair(idx)?;
            }
        }

        // Build the report in the documented row order, omitting count==0.
        let mut stages: Vec<StageStats> = Vec::new();
        let kernel_order: [KernelStage; 9] = [
            KernelStage::VvKickDrift,
            KernelStage::VvKickDriftLossless,
            KernelStage::LangevinKickHalf,
            KernelStage::LangevinDriftHalf,
            KernelStage::LangevinOuStep,
            KernelStage::LjPairForce,
            KernelStage::ReducePairForces,
            KernelStage::VvKick,
            KernelStage::VvKickLossless,
        ];
        let host_order: [HostStage; 9] = [
            HostStage::HostToDeviceUpload,
            HostStage::DeviceToHostDownload,
            HostStage::TrajectoryWrite,
            HostStage::LogWrite,
            HostStage::VelocityGeneration,
            HostStage::ConfigLoad,
            HostStage::InitLoad,
            HostStage::GpuInit,
            HostStage::TotalRuntime,
        ];

        for k in kernel_order {
            let acc = self.kernel_acc[k.index()];
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
        for h in host_order {
            let acc = self.host_acc[h.index()];
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
        for i in 0..9 {
            unsafe {
                let _ = event::destroy(self.kernel_starts[i]);
                let _ = event::destroy(self.kernel_stops[i]);
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
