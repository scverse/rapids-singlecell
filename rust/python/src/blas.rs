//! Native cuBLAS calls using the library already selected by CuPy.
//!
//! Resolving symbols through CuPy's imported extension preserves its CUDA 12/13
//! dependency selection. No toolkit link or independently loaded cuBLAS version
//! is needed, and native stream capture retains the original backend behavior.
#![allow(clippy::too_many_arguments)]

use crate::array::Dtype;
use libloading::Library;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{ffi::c_void, sync::OnceLock};

type Handle = *mut c_void;
type SetStream = unsafe extern "C" fn(Handle, *mut c_void) -> i32;
type GetPointerMode = unsafe extern "C" fn(Handle, *mut i32) -> i32;
type SetPointerMode = unsafe extern "C" fn(Handle, i32) -> i32;
type Gemm<T> = unsafe extern "C" fn(
    Handle,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const T,
    *const T,
    i32,
    *const T,
    i32,
    *const T,
    *mut T,
    i32,
) -> i32;

struct Blas {
    // Keep the imported extension and its exact cuBLAS dependency loaded for
    // the lifetime of every cached function pointer.
    _library: Library,
    set_stream: SetStream,
    get_pointer_mode: GetPointerMode,
    set_pointer_mode: SetPointerMode,
    sgemm: Gemm<f32>,
    dgemm: Gemm<f64>,
}

impl Blas {
    fn get(py: Python<'_>) -> PyResult<&'static Self> {
        static BLAS: OnceLock<Blas> = OnceLock::new();
        if let Some(api) = BLAS.get() {
            return Ok(api);
        }
        let extension = py.import("cupy_backends.cuda.libs.cublas")?;
        let path = extension.getattr("__file__")?.extract::<String>()?;
        // SAFETY: Python has already imported this extension. Reopening the
        // same file retains it; dlsym searches that handle's dependency tree.
        // These types match the stable cuBLAS v2 C ABI in cublas_api.h.
        let api = unsafe {
            let library = Library::new(path).map_err(load_error)?;
            Self {
                set_stream: *library.get(b"cublasSetStream_v2\0").map_err(load_error)?,
                get_pointer_mode: *library
                    .get(b"cublasGetPointerMode_v2\0")
                    .map_err(load_error)?,
                set_pointer_mode: *library
                    .get(b"cublasSetPointerMode_v2\0")
                    .map_err(load_error)?,
                sgemm: *library.get(b"cublasSgemm_v2\0").map_err(load_error)?,
                dgemm: *library.get(b"cublasDgemm_v2\0").map_err(load_error)?,
                _library: library,
            }
        };
        // Imports happen outside OnceLock initialization, so Python reentrancy
        // cannot deadlock its initializer. A concurrent spare handle can drop.
        Ok(BLAS.get_or_init(|| api))
    }
}

fn load_error(error: libloading::Error) -> PyErr {
    PyRuntimeError::new_err(format!("cannot resolve CuPy's cuBLAS library: {error}"))
}

fn check(status: i32, operation: &str) -> PyResult<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(format!(
            "{operation} failed with cuBLAS status {status}"
        )))
    }
}

struct HostScalars<'a> {
    api: &'a Blas,
    handle: Handle,
    original_mode: i32,
    restore: bool,
}

impl<'a> HostScalars<'a> {
    unsafe fn new(api: &'a Blas, handle: Handle) -> PyResult<Self> {
        let mut original_mode = 0;
        // SAFETY: the caller borrows a live cuBLAS handle; mode is a host slot.
        check(
            unsafe { (api.get_pointer_mode)(handle, &mut original_mode) },
            "cublasGetPointerMode",
        )?;
        if original_mode != 0 {
            check(
                unsafe { (api.set_pointer_mode)(handle, 0) },
                "cublasSetPointerMode",
            )?;
        }
        Ok(Self {
            api,
            handle,
            original_mode,
            restore: original_mode != 0,
        })
    }

    fn finish(mut self) -> PyResult<()> {
        if self.restore {
            // SAFETY: this guard still borrows the same live handle.
            check(
                unsafe { (self.api.set_pointer_mode)(self.handle, self.original_mode) },
                "cublasSetPointerMode restore",
            )?;
            self.restore = false;
        }
        Ok(())
    }
}

impl Drop for HostScalars<'_> {
    fn drop(&mut self) {
        if self.restore {
            // SAFETY: restore borrowed handle state also on cuBLAS errors.
            unsafe { (self.api.set_pointer_mode)(self.handle, self.original_mode) };
        }
    }
}

/// Enqueue GEMM on a borrowed cuBLAS handle with host alpha/beta scalars.
/// Callers validate device allocation extents, disjoint outputs and the current
/// device, and retain inputs/outputs/handle through stream completion.
pub(crate) fn gemm(
    py: Python<'_>,
    dtype: Dtype,
    handle: usize,
    stream: usize,
    ta: i32,
    tb: i32,
    m: u64,
    n: u64,
    k: u64,
    a: u64,
    lda: u64,
    b: u64,
    ldb: u64,
    c: u64,
    ldc: u64,
    beta: f64,
) -> PyResult<()> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    if handle == 0 {
        return Err(PyValueError::new_err("cuBLAS handle must be non-null"));
    }
    if !matches!(dtype, Dtype::F32 | Dtype::F64) {
        return Err(PyValueError::new_err(
            "cuBLAS GEMM requires float32 or float64",
        ));
    }
    for d in [m, n, k, lda, ldb, ldc] {
        if d > i32::MAX as u64 {
            return Err(PyValueError::new_err("cuBLAS dimensions exceed int32"));
        }
    }
    let api = Blas::get(py)?;
    let handle = handle as Handle;
    // SAFETY: callers have validated the matrix shapes, strides, device and
    // allocation extents. cuBLAS copies host scalars before returning, including
    // during graph capture; device allocations remain owned by Python callers.
    let scalars = unsafe { HostScalars::new(api, handle) }?;
    check(
        unsafe { (api.set_stream)(handle, stream as *mut c_void) },
        "cublasSetStream",
    )?;
    let status = unsafe {
        if dtype == Dtype::F32 {
            (api.sgemm)(
                handle,
                ta,
                tb,
                m as i32,
                n as i32,
                k as i32,
                &1.0_f32,
                a as *const f32,
                lda as i32,
                b as *const f32,
                ldb as i32,
                &(beta as f32),
                c as *mut f32,
                ldc as i32,
            )
        } else {
            (api.dgemm)(
                handle,
                ta,
                tb,
                m as i32,
                n as i32,
                k as i32,
                &1.0_f64,
                a as *const f64,
                lda as i32,
                b as *const f64,
                ldb as i32,
                &beta,
                c as *mut f64,
                ldc as i32,
            )
        }
    };
    check(status, "cuBLAS GEMM")?;
    scalars.finish()
}
