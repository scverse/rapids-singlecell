//! Typed PyO3 entry points for domain-specific CUDA kernels.
use crate::{
    array::{Array, Dtype, Layout, current_device},
    runtime,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use std::ffi::c_void;

struct Buffer {
    array: Option<Array>,
    order: u64,
    rows: u64,
    cols: u64,
}
impl Buffer {
    fn pointer(&self) -> u64 {
        self.array.as_ref().map_or(0, |a| a.pointer)
    }
    fn len(&self) -> u64 {
        self.array.as_ref().map_or(0, |a| a.len)
    }
    fn kind(&self) -> u64 {
        self.array.as_ref().map_or(0, |a| match a.dtype {
            Dtype::F32 => 0,
            Dtype::F64 => 1,
            Dtype::I32 => 2,
            Dtype::I64 => 3,
            Dtype::Bool => 4,
            Dtype::U32 => 5,
            Dtype::U64 => 6,
            _ => 99,
        })
    }
}
fn read(
    object: Option<&Bound<'_, PyAny>>,
    cupy: &Bound<'_, PyModule>,
    name: &str,
    expected: &str,
    contiguous: bool,
) -> PyResult<Buffer> {
    let Some(object) = object else {
        return Ok(Buffer {
            array: None,
            order: 0,
            rows: 0,
            cols: 0,
        });
    };
    let array = Array::read(
        object,
        cupy,
        name,
        None,
        None,
        if contiguous {
            Layout::Contiguous
        } else {
            Layout::C
        },
    )?;
    let accepted = match expected {
        "Index" => matches!(
            array.dtype,
            Dtype::I32 | Dtype::I64 | Dtype::U32 | Dtype::U64
        ),
        "T" => matches!(array.dtype, Dtype::F32 | Dtype::F64),
        "IdxT" | "AdjIdxT" | "DataIdxT" | "IndptrT" | "OutIdxT" => {
            matches!(array.dtype, Dtype::I32 | Dtype::I64)
        }
        "float" => array.dtype == Dtype::F32,
        "double" => array.dtype == Dtype::F64,
        "int" => array.dtype == Dtype::I32,
        "int64_t" => array.dtype == Dtype::I64,
        "unsigned int" => array.dtype == Dtype::U32,
        "cooc_count_t" => array.dtype == Dtype::U64,
        "bool" => array.dtype == Dtype::Bool,
        _ => false,
    };
    if !accepted {
        return Err(PyTypeError::new_err(format!(
            "unsupported {name} dtype: {}",
            array.dtype.name()
        )));
    }
    let order = u64::from(
        !object
            .getattr("flags")?
            .getattr("c_contiguous")?
            .extract::<bool>()?,
    );
    let rows = array.shape.first().copied().unwrap_or(0) as u64;
    let cols = array.shape.get(1).copied().unwrap_or(1) as u64;
    Ok(Buffer {
        array: Some(array),
        order,
        rows,
        cols,
    })
}
fn same(a: &Buffer, b: &Buffer) -> PyResult<()> {
    if a.array.is_some() && b.array.is_some() && a.kind() != b.kind() {
        return Err(PyTypeError::new_err(
            "arrays of the same value or index type must have matching dtypes",
        ));
    }
    Ok(())
}
fn launch(
    cupy: &Bound<'_, PyModule>,
    buffers: &[&Buffer],
    name: &str,
    work: u64,
    stream: usize,
    mut args: Vec<u64>,
) -> PyResult<()> {
    let arrays: Vec<&Array> = buffers.iter().filter_map(|b| b.array.as_ref()).collect();
    let device = current_device(cupy, &arrays)?;
    if work == 0 {
        return Ok(());
    }
    for b in buffers {
        args.extend([b.len(), b.kind(), b.order, b.rows, b.cols]);
    }
    let block = match name {
        "domain_ligrec_sum_count_dense" | "domain_ligrec_mean_dense" => 1024,
        "domain_guide_assignment_assign_threshold_dense"
        | "domain_guide_assignment_fit_assign_dense"
        | "domain_mixscale_project_score" => 256,
        _ => 128,
    };
    let mut pointers: Vec<*mut c_void> = args.iter_mut().map(|a| (a as *mut u64).cast()).collect();
    // SAFETY: every argument is a u64 slot, matching the device entry ABI;
    // descriptors are validated CuPy allocations. Device Buffer bounds-checks
    // all accesses, including those indexed by sparse metadata.
    unsafe {
        runtime::launch(
            device,
            name,
            (work.div_ceil(block).min(65_535) as u32, 1, 1),
            (block as u32, 1, 1),
            stream,
            &mut pointers,
        )
    }
}
fn submodule<'py>(parent: &Bound<'py, PyModule>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    let py = parent.py();
    let full = format!("rapids_singlecell._cuda.{name}");
    let module = PyModule::new(py, &full)?;
    module.setattr("__backend__", "rust")?;
    parent.add_submodule(&module)?;
    parent.setattr(name, &module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item(full, &module)?;
    Ok(module)
}
// Domain module registry.
mod aggr;
mod aucell;
mod autocorr;
mod bbknn;
mod cooc;
mod edistance;
mod guide_assignment;
mod kde;
mod ligrec;
mod mixscale;
mod nn_descent;
mod pseudobulk;
mod pv;
mod sinkhorn;
mod spca;
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    aggr::register(parent)?;
    pseudobulk::register(parent)?;
    autocorr::register(parent)?;
    cooc::register(parent)?;
    ligrec::register(parent)?;
    kde::register(parent)?;
    edistance::register(parent)?;
    sinkhorn::register(parent)?;
    guide_assignment::register(parent)?;
    mixscale::register(parent)?;
    pv::register(parent)?;
    aucell::register(parent)?;
    bbknn::register(parent)?;
    nn_descent::register(parent)?;
    spca::register(parent)?;
    Ok(())
}
