//! Bindings for array primitives shared by preprocessing, aggregation and SVD.
use crate::{
    array::{Dtype, Layout},
    preprocessing::Launch,
    runtime,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

#[pyfunction]
#[pyo3(signature = (alpha, y, x, *, stream=0))]
fn axpy(
    py: Python<'_>,
    alpha: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let x = call.data(x, None, true, false, true)?;
    let len = call.len();
    let y = call.vector(y, "y", call.value, Some(len), false)?;
    let a = call.array(
        alpha,
        "alpha",
        Some(call.value),
        Some(1),
        false,
        Layout::C,
        false,
    )?;
    call.run("axpy", len, 256, false, stream, vec![a, y, x, len])
}

#[pyfunction]
#[pyo3(signature = (data, output, *, axis, square, stream=0))]
fn dense_sum(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    axis: u32,
    square: bool,
    stream: usize,
) -> PyResult<()> {
    if axis > 1 {
        return Err(PyValueError::new_err("axis must be 0 or 1"));
    }
    let shape = data.getattr("shape")?.extract::<Vec<u64>>()?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err("data must be two-dimensional"));
    }
    let mut call = Launch::new(py)?;
    let fortran = !data
        .getattr("flags")?
        .getattr("c_contiguous")?
        .extract::<bool>()?;
    let d = call.data(data, None, false, fortran, false)?;
    let major = shape[1 - axis as usize];
    let o = call.vector(output, "output", Dtype::F64, Some(major), true)?;
    let strides = data.getattr("strides")?.extract::<Vec<u64>>()?;
    let bytes = if call.value == Dtype::F32 { 4 } else { 8 };
    let row_stride = strides[0] / bytes;
    let col_stride = strides[1] / bytes;
    let reduction_stride = if axis == 0 { row_stride } else { col_stride };
    let minor = shape[axis as usize];
    // Partition long strided reductions so the device has enough independent
    // warps, then combine the short, contiguous partial-output matrix.
    let parts = if reduction_stride != 1 && major != 0 {
        minor.div_ceil(256).clamp(1, 256)
    } else {
        1
    };
    let cupy = py.import("cupy")?;
    let _scope = if parts > 1 {
        Some(runtime::StreamScope::new(&cupy, stream)?)
    } else {
        None
    };
    let scratch = if parts > 1 {
        Some(cupy.call_method1("empty", ((parts, major), "float64"))?)
    } else {
        None
    };
    let target = if let Some(scratch) = &scratch {
        call.array(
            scratch,
            "partial sums",
            Some(Dtype::F64),
            None,
            false,
            Layout::C,
            true,
        )?
    } else {
        o
    };
    call.run(
        "dense_sum",
        major * parts,
        32,
        reduction_stride == 1,
        stream,
        vec![
            d,
            target,
            shape[0],
            shape[1],
            row_stride,
            col_stride,
            axis as u64,
            u64::from(square),
            parts,
        ],
    )?;
    if let Some(scratch) = &scratch {
        let mut finish = Launch::new(py)?;
        let input = finish.data(scratch, None, false, false, false)?;
        let output = finish.vector(output, "output", Dtype::F64, Some(major), true)?;
        finish.run(
            "dense_sum",
            major,
            32,
            false,
            stream,
            vec![input, output, parts, major, major, 1, 0, 0, 1],
        )?;
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (data, indices, clip, squares, sums, *, stream=0))]
fn clip_sums(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    clip: &Bound<'_, PyAny>,
    squares: &Bound<'_, PyAny>,
    sums: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    let len = call.len();
    let i = call.indices(indices, len)?;
    let c = call.vector(clip, "clip", Dtype::F64, None, false)?;
    let genes = call.len();
    let sq = call.vector(squares, "squares", Dtype::F64, Some(genes), true)?;
    let s = call.vector(sums, "sums", Dtype::F64, Some(genes), true)?;
    call.run(
        "clip_sums",
        len,
        256,
        false,
        stream,
        vec![d, i, c, sq, s, len, genes],
    )
}

#[pyfunction]
#[pyo3(signature = (indices, counts, *, stream=0))]
fn count_indices(
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    counts: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let len = indices.getattr("size")?.extract()?;
    let i = call.indices(indices, len)?;
    let c = call.vector(counts, "counts", Dtype::I32, None, true)?;
    let genes = call.len();
    call.run("count", len, 256, false, stream, vec![i, c, len, genes])
}

#[pyfunction]
#[pyo3(signature = (rows, cols, output, *, stream=0))]
fn duplicates_diff(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    cols: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let len = rows.getattr("size")?.extract()?;
    let r = call.indices(rows, len)?;
    let c = call.indices(cols, len)?;
    let o = call.vector(output, "output", call.index.unwrap(), Some(len), true)?;
    call.run(
        "duplicates_diff",
        len,
        256,
        false,
        stream,
        vec![r, c, o, len],
    )
}

#[pyfunction]
#[pyo3(signature = (src_rows, src_cols, indices, rows, cols, *, stream=0))]
fn duplicates_assign(
    py: Python<'_>,
    src_rows: &Bound<'_, PyAny>,
    src_cols: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    rows: &Bound<'_, PyAny>,
    cols: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let len = src_rows.getattr("size")?.extract()?;
    let sr = call.indices(src_rows, len)?;
    let sc = call.indices(src_cols, len)?;
    let i = call.indices(indices, len)?;
    let r = call.vector(rows, "rows", call.index.unwrap(), None, true)?;
    let groups = call.len();
    let c = call.vector(cols, "cols", call.index.unwrap(), Some(groups), true)?;
    call.run(
        "duplicates_assign",
        len,
        256,
        false,
        stream,
        vec![sr, sc, i, r, c, len, groups],
    )
}

#[pyfunction]
#[pyo3(signature = (data, indices, output, *, squares=None, count=false, stream=0))]
fn scatter(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    squares: Option<&Bound<'_, PyAny>>,
    count: bool,
    stream: usize,
) -> PyResult<()> {
    let mut call = Launch::new(py)?;
    let d = call.data(data, None, true, false, false)?;
    if call.value != Dtype::F64 {
        return Err(PyTypeError::new_err("data must have dtype float64"));
    }
    if count && squares.is_some() {
        return Err(PyValueError::new_err(
            "count and squares cannot be combined",
        ));
    }
    let len = call.len();
    let i = call.indices(indices, len)?;
    let o = call.vector(
        output,
        "output",
        if count { Dtype::F32 } else { Dtype::F64 },
        None,
        true,
    )?;
    let groups = call.len();
    let sq = match squares {
        Some(s) => call.vector(s, "squares", Dtype::F64, Some(groups), true)?,
        None => 0,
    };
    let mode = if count {
        2
    } else {
        u64::from(squares.is_some())
    };
    call.run(
        "scatter",
        len,
        256,
        false,
        stream,
        vec![
            d,
            i,
            if count { 0 } else { o },
            sq,
            if count { o } else { 0 },
            len,
            groups,
            mode,
        ],
    )
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "rapids_singlecell._cuda._elementwise_cuda")?;
    module.setattr("__backend__", "rust")?;
    module.add_function(wrap_pyfunction!(axpy, &module)?)?;
    module.add_function(wrap_pyfunction!(dense_sum, &module)?)?;
    module.add_function(wrap_pyfunction!(clip_sums, &module)?)?;
    module.add_function(wrap_pyfunction!(count_indices, &module)?)?;
    module.add_function(wrap_pyfunction!(duplicates_diff, &module)?)?;
    module.add_function(wrap_pyfunction!(duplicates_assign, &module)?)?;
    module.add_function(wrap_pyfunction!(scatter, &module)?)?;
    parent.add_submodule(&module)?;
    Ok(())
}
