// CUDA graph safe wrappers. cudarc 0.13.9 exposes only the raw `sys`
// bindings for graph capture, so this module is a thin RAII layer over
// the unsafe `cuStreamBeginCapture_v2`, `cuStreamEndCapture`,
// `cuGraphInstantiateWithFlags`, `cuGraphLaunch`, `cuGraphDestroy`, and
// `cuGraphExecDestroy` entry points. See `cuda-graphs.md` for the
// runtime design.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, DriverError};
use cudarc::driver::sys;

/// Stream-capture mode passed to `begin_stream_capture`. Mirrors the
/// CUDA `CUstreamCaptureMode` enum.
#[derive(Debug, Copy, Clone)]
pub enum CaptureMode {
    Global,
    ThreadLocal,
    Relaxed,
}

impl CaptureMode {
    fn to_sys(self) -> sys::CUstreamCaptureMode {
        match self {
            CaptureMode::Global => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
            CaptureMode::ThreadLocal => {
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            }
            CaptureMode::Relaxed => sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

/// Errors surfaced by the CUDA-graph wrapper.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("cuStreamBeginCapture_v2 failed: {0:?}")]
    BeginCaptureFailed(DriverError),
    #[error("cuStreamEndCapture failed: {0:?}")]
    EndCaptureFailed(DriverError),
    #[error("cuGraphInstantiateWithFlags failed: {0:?}")]
    InstantiateFailed(DriverError),
    #[error("cuGraphLaunch failed: {0:?}")]
    LaunchFailed(DriverError),
    #[error("cuGraphDestroy failed: {0:?}")]
    DestroyFailed(DriverError),
}

/// Owned, raw-pointer-shaped CUDA graph handle. Drop calls
/// `cuGraphDestroy`. Construct via `end_stream_capture`.
#[derive(Debug)]
pub struct CudaGraph {
    handle: sys::CUgraph,
    device: Arc<CudaDevice>,
}

/// Per-node-type counts from a captured `CudaGraph`. Diagnostic only —
/// used to verify whether secondary-stream work was actually
/// incorporated into a captured graph.
#[derive(Debug, Default, Clone, Copy)]
pub struct GraphNodeSummary {
    pub kernel: usize,
    pub memcpy: usize,
    pub memset: usize,
    pub host: usize,
    pub child_graph: usize,
    pub empty: usize,
    pub wait_event: usize,
    pub event_record: usize,
    pub other: usize,
    pub total: usize,
}

impl CudaGraph {
    /// Counts the nodes in this graph by type. Used to verify whether
    /// secondary-stream work (e.g. SPME recip's cuFFT and influence
    /// kernels) was incorporated into a captured default-stream graph
    /// via cross-stream event coordination. A graph that "looks too
    /// small" is evidence the secondary stream was never pulled into
    /// capture.
    pub fn node_summary(&self) -> Result<GraphNodeSummary, GraphError> {
        self.device
            .bind_to_thread()
            .map_err(GraphError::InstantiateFailed)?;
        let mut num: usize = 0;
        unsafe {
            sys::lib()
                .cuGraphGetNodes
                .as_ref()
                .expect("cuGraphGetNodes symbol unavailable")(
                self.handle,
                std::ptr::null_mut(),
                &mut num,
            )
            .result()
            .map_err(GraphError::InstantiateFailed)?;
        }
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); num];
        unsafe {
            sys::lib()
                .cuGraphGetNodes
                .as_ref()
                .expect("cuGraphGetNodes symbol unavailable")(
                self.handle,
                nodes.as_mut_ptr(),
                &mut num,
            )
            .result()
            .map_err(GraphError::InstantiateFailed)?;
        }
        let mut summary = GraphNodeSummary::default();
        for node in &nodes {
            let mut ty: sys::CUgraphNodeType =
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            unsafe {
                sys::lib()
                    .cuGraphNodeGetType
                    .as_ref()
                    .expect("cuGraphNodeGetType symbol unavailable")(
                    *node, &mut ty
                )
                .result()
                .map_err(GraphError::InstantiateFailed)?;
            }
            summary.total += 1;
            match ty {
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => summary.kernel += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => summary.memcpy += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => summary.memset += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => summary.host += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH => summary.child_graph += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => summary.empty += 1,
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT => {
                    summary.wait_event += 1
                }
                sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD => {
                    summary.event_record += 1
                }
                _ => summary.other += 1,
            }
        }
        Ok(summary)
    }

    /// Instantiate this graph into an executable graph. Wraps
    /// `cuGraphInstantiateWithFlags` with `flags = 0`.
    pub fn instantiate(&self) -> Result<CudaGraphExec, GraphError> {
        self.device
            .bind_to_thread()
            .map_err(GraphError::InstantiateFailed)?;
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let result = unsafe {
            sys::lib().cuGraphInstantiateWithFlags.as_ref().expect(
                "cuGraphInstantiateWithFlags symbol unavailable; CUDA driver too old",
            )(&mut exec, self.handle, 0)
        };
        result
            .result()
            .map_err(GraphError::InstantiateFailed)?;
        Ok(CudaGraphExec {
            handle: exec,
            device: self.device.clone(),
        })
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // Best-effort destroy: nothing to do on error; the next CUDA
        // call will surface any latent problem.
        if let Err(_e) = self.device.bind_to_thread() {
            return;
        }
        let _ = unsafe {
            sys::lib()
                .cuGraphDestroy
                .as_ref()
                .expect("cuGraphDestroy symbol unavailable")(self.handle)
                .result()
        };
    }
}

/// Owned, instantiated CUDA graph. Drop calls `cuGraphExecDestroy`.
/// Construct via `CudaGraph::instantiate`.
#[derive(Debug)]
pub struct CudaGraphExec {
    handle: sys::CUgraphExec,
    device: Arc<CudaDevice>,
}

impl CudaGraphExec {
    /// Launch the graph on the device's default stream. Wraps
    /// `cuGraphLaunch`. The call is asynchronous from the host's
    /// perspective; the caller is responsible for any subsequent
    /// `synchronize()` it needs.
    pub fn launch(&self) -> Result<(), GraphError> {
        self.device
            .bind_to_thread()
            .map_err(GraphError::LaunchFailed)?;
        let stream = *self.device.cu_stream();
        let result = unsafe {
            sys::lib()
                .cuGraphLaunch
                .as_ref()
                .expect("cuGraphLaunch symbol unavailable")(self.handle, stream)
        };
        result.result().map_err(GraphError::LaunchFailed)
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Err(_e) = self.device.bind_to_thread() {
            return;
        }
        let _ = unsafe {
            sys::lib()
                .cuGraphExecDestroy
                .as_ref()
                .expect("cuGraphExecDestroy symbol unavailable")(self.handle)
                .result()
        };
    }
}

/// Begins stream capture on `device`'s default stream. Wraps
/// `cuStreamBeginCapture_v2`. Every CUDA kernel launch on that stream
/// (and on any stream that joins it via events) between this call and
/// `end_stream_capture` is recorded as a node in the captured graph.
pub fn begin_stream_capture(
    device: &Arc<CudaDevice>,
    mode: CaptureMode,
) -> Result<(), GraphError> {
    device
        .bind_to_thread()
        .map_err(GraphError::BeginCaptureFailed)?;
    let stream = *device.cu_stream();
    let result = unsafe {
        sys::lib()
            .cuStreamBeginCapture_v2
            .as_ref()
            .expect("cuStreamBeginCapture_v2 symbol unavailable")(stream, mode.to_sys())
    };
    result.result().map_err(GraphError::BeginCaptureFailed)
}

/// Ends stream capture on `device`'s default stream and returns the
/// captured graph. Wraps `cuStreamEndCapture`. The captured graph is
/// not yet executable — call `CudaGraph::instantiate` to obtain an
/// executable.
pub fn end_stream_capture(device: &Arc<CudaDevice>) -> Result<CudaGraph, GraphError> {
    device
        .bind_to_thread()
        .map_err(GraphError::EndCaptureFailed)?;
    let stream = *device.cu_stream();
    let mut handle: sys::CUgraph = std::ptr::null_mut();
    let result = unsafe {
        sys::lib()
            .cuStreamEndCapture
            .as_ref()
            .expect("cuStreamEndCapture symbol unavailable")(stream, &mut handle)
    };
    result.result().map_err(GraphError::EndCaptureFailed)?;
    Ok(CudaGraph {
        handle,
        device: device.clone(),
    })
}
