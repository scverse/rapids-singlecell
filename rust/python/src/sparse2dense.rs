use crate::{
    array::{Array, Dtype, Layout, current_device},
    runtime,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use std::ffi::c_void;

/// Accumulate CSR/CSC values into contiguous output on a borrowed CUDA stream.
/// Arrays and stream must remain alive until the asynchronous work completes.
#[pyfunction]
#[pyo3(signature = (indptr, index, data, *, out, major, minor, c_switch, max_nnz, stream=0))]
#[allow(clippy::too_many_arguments)] // Preserve the existing Python extension API.
pub fn sparse2dense(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    major: i64,
    minor: i64,
    c_switch: bool,
    max_nnz: i64,
    stream: usize,
) -> PyResult<()> {
    if major < 0 || minor < 0 || max_nnz < 0 {
        return Err(PyValueError::new_err(
            "major, minor and max_nnz must be nonnegative",
        ));
    }
    let mut major = major as u64;
    let mut minor = minor as u64;
    let out_len = major
        .checked_mul(minor)
        .filter(|&n| n <= isize::MAX as u64 / 4)
        .ok_or_else(|| PyValueError::new_err("output size exceeds addressable memory"))?;
    let cupy = py.import("cupy")?;
    let rows = Array::read(
        indptr,
        &cupy,
        "indptr",
        None,
        Some(&[major as usize + 1]),
        Layout::C,
    )?;
    if !matches!(rows.dtype, Dtype::I32 | Dtype::I64) {
        return Err(PyTypeError::new_err(
            "indptr must have dtype int32 or int64",
        ));
    }
    let indices = Array::read(index, &cupy, "index", Some(rows.dtype), None, Layout::C)?;
    indices.require_vector("index")?;
    let values = Array::read(
        data,
        &cupy,
        "data",
        None,
        Some(&[indices.len as usize]),
        Layout::C,
    )?;
    if !matches!(values.dtype, Dtype::F32 | Dtype::F64) {
        return Err(PyTypeError::new_err(
            "data must have dtype float32 or float64",
        ));
    }
    let output = Array::read(
        out,
        &cupy,
        "out",
        Some(values.dtype),
        None,
        Layout::Contiguous,
    )?;
    if output.len != out_len {
        return Err(PyValueError::new_err("out size must equal major * minor"));
    }
    let device = current_device(&cupy, &[&rows, &indices, &values, &output])?;
    for input in [&rows, &indices, &values] {
        output.require_disjoint(input)?;
    }
    if major == 0 || minor == 0 || max_nnz == 0 || indices.len == 0 {
        return Ok(());
    }
    let name = match (values.dtype, rows.dtype) {
        (Dtype::F32, Dtype::I32) => "sparse2dense_f32_i32",
        (Dtype::F32, Dtype::I64) => "sparse2dense_f32_i64",
        (Dtype::F64, Dtype::I32) => "sparse2dense_f64_i32",
        (Dtype::F64, Dtype::I64) => "sparse2dense_f64_i64",
        _ => unreachable!("dtypes were validated above"),
    };
    let mut row_pointer = rows.pointer;
    let mut index_pointer = indices.pointer;
    let mut data_pointer = values.pointer;
    let mut output_pointer = output.pointer;
    let mut nnz = indices.len;
    let mut c_order = u32::from(c_switch);
    let mut arguments = [
        (&mut row_pointer as *mut u64).cast::<c_void>(),
        (&mut index_pointer as *mut u64).cast::<c_void>(),
        (&mut data_pointer as *mut u64).cast::<c_void>(),
        (&mut output_pointer as *mut u64).cast::<c_void>(),
        (&mut major as *mut u64).cast::<c_void>(),
        (&mut minor as *mut u64).cast::<c_void>(),
        (&mut nnz as *mut u64).cast::<c_void>(),
        (&mut c_order as *mut u32).cast::<c_void>(),
    ];
    // SAFETY: argument types match the selected specialization. Metadata proves
    // allocation bounds/dtypes/layout; device code checks row spans/indices. All
    // output writes are atomic and output is disjoint from the borrowed inputs.
    unsafe {
        runtime::launch(
            device,
            name,
            (
                major.div_ceil(32).min(65_535) as u32,
                (max_nnz as u64).div_ceil(32).min(65_535) as u32,
                1,
            ),
            (32, 32, 1),
            stream,
            &mut arguments,
        )
    }
}
