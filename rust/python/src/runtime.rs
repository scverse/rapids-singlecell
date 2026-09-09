//! CUDA module caching and launches borrowing the caller's context and stream.

use cuda_core::{CudaContext, CudaFunction, CudaModule, DriverError, IntoResult, sys};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{
    collections::HashMap,
    ffi::c_void,
    ptr,
    sync::{Arc, Mutex, OnceLock},
};

const PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// Run CuPy allocations and primitives on the stream borrowed by a binding.
/// The context manager restores the caller's stream on every return path.
#[pyclass(frozen)]
struct BorrowedStream {
    pointer: usize,
}

#[pymethods]
impl BorrowedStream {
    /// CUDA stream protocol, version 0. Ownership stays with the caller.
    fn __cuda_stream__(&self) -> (u32, usize) {
        (0, self.pointer)
    }
}

pub struct StreamScope<'py>(Bound<'py, PyAny>);

impl<'py> StreamScope<'py> {
    pub fn new(cupy: &Bound<'py, PyModule>, stream: usize) -> PyResult<Self> {
        let cuda = cupy.getattr("cuda")?;
        let stream_type = cuda.getattr("Stream")?;
        let context = if stream == 0 {
            stream_type.getattr("null")?
        } else if stream_type.hasattr("from_external")? {
            let borrowed = Py::new(cupy.py(), BorrowedStream { pointer: stream })?;
            stream_type.call_method1("from_external", (borrowed,))?
        } else {
            cuda.getattr("ExternalStream")?.call1((stream,))?
        };
        context.call_method0("__enter__")?;
        Ok(Self(context))
    }
}

impl Drop for StreamScope<'_> {
    fn drop(&mut self) {
        let py = self.0.py();
        let _ = self
            .0
            .call_method1("__exit__", (py.None(), py.None(), py.None()));
    }
}
struct DeviceModule {
    context: Arc<CudaContext>,
    module: Arc<CudaModule>,
    functions: Mutex<HashMap<String, CudaFunction>>,
}

// No Python objects are stored here. Each function retains the module/context,
// keeping the PTX loaded until all asynchronous work has completed.
static MODULES: OnceLock<Mutex<HashMap<usize, Arc<DeviceModule>>>> = OnceLock::new();

fn cuda_error(error: DriverError) -> PyErr {
    PyRuntimeError::new_err(format!("Rust CUDA backend: {error}"))
}

struct ContextGuard(sys::CUcontext);

impl ContextGuard {
    fn capture() -> PyResult<Self> {
        let mut context = ptr::null_mut();
        // SAFETY: context is a valid host output slot.
        unsafe { sys::cuCtxGetCurrent(&mut context) }
            .result()
            .map_err(cuda_error)?;
        Ok(Self(context))
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        // SAFETY: this borrowed context was current on this thread at entry.
        unsafe { sys::cuCtxSetCurrent(self.0) };
    }
}

fn device_module(device: usize) -> PyResult<Arc<DeviceModule>> {
    let mut modules = MODULES
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Rust CUDA module cache lock poisoned"))?;
    if let Some(module) = modules.get(&device) {
        return Ok(Arc::clone(module));
    }
    let context = CudaContext::new(device).map_err(cuda_error)?;
    let module = context.load_module_from_image(PTX).map_err(cuda_error)?;
    let module = Arc::new(DeviceModule {
        context,
        module,
        functions: Mutex::new(HashMap::new()),
    });
    modules.insert(device, Arc::clone(&module));
    Ok(module)
}

/// Enqueue a native kernel without constructing owners for foreign allocations.
///
/// # Safety
/// Arguments must match the named PTX entry's ABI and refer to validated device
/// allocations. The caller must keep allocations/stream alive until completion,
/// order cross-stream access, and choose grid/block sizes required by the kernel.
pub unsafe fn launch(
    device: usize,
    name: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    stream: usize,
    arguments: &mut [*mut c_void],
) -> PyResult<()> {
    let caller_context = ContextGuard::capture()?;
    let module = device_module(device)?;
    if !caller_context.0.is_null() && caller_context.0 != module.context.cu_ctx() {
        return Err(PyValueError::new_err(
            "Rust CUDA kernels require the current device's primary context",
        ));
    }
    let mut functions = module
        .functions
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Rust CUDA function cache lock poisoned"))?;
    if !functions.contains_key(name) {
        functions.insert(
            name.to_owned(),
            module.module.load_function(name).map_err(cuda_error)?,
        );
    }
    let function = functions
        .get(name)
        .expect("function inserted above")
        .clone();
    drop(functions);
    module.context.bind_to_thread().map_err(cuda_error)?;
    // SAFETY: the caller establishes kernel-specific ABI and memory invariants.
    // CUDA copies argument slots during this call; device pointers stay borrowed.
    unsafe {
        cuda_core::launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            stream as sys::CUstream,
            arguments,
        )
    }
    .map_err(cuda_error)
}
