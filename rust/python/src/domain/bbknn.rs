//! bbknn bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(data, indptr, *, n_rows, trim, vals, stream=0))]
fn find_top_k_per_row(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    n_rows: u64,
    trim: u64,
    vals: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    if trim == 0 {
        return Err(PyValueError::new_err("trim must be positive"));
    }
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "float", false)?;
    let indptr = read(Some(indptr), &cupy, "indptr", "int", false)?;
    let vals = read(Some(vals), &cupy, "vals", "float", false)?;
    launch(
        &cupy,
        &[&data, &indptr, &vals],
        "domain_bbknn_find_top_k_per_row",
        n_rows * 32,
        stream,
        vec![
            data.pointer(),
            indptr.pointer(),
            n_rows,
            trim,
            vals.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(data, indptr, *, n_rows, trim, vals, stream=0))]
fn find_top_k_per_row_sorted(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    n_rows: u64,
    trim: u64,
    vals: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    if trim == 0 {
        return Err(PyValueError::new_err("trim must be positive"));
    }
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "float", false)?;
    let indptr = read(Some(indptr), &cupy, "indptr", "int", false)?;
    let vals = read(Some(vals), &cupy, "vals", "float", false)?;
    launch(
        &cupy,
        &[&data, &indptr, &vals],
        "domain_bbknn_find_top_k_per_row_sorted",
        n_rows * 32,
        stream,
        vec![
            data.pointer(),
            indptr.pointer(),
            n_rows,
            trim,
            vals.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(indptr, index, data, *, vals, n_rows, stream=0))]
fn cut_smaller(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    vals: &Bound<'_, PyAny>,
    n_rows: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let indptr = read(Some(indptr), &cupy, "indptr", "int", false)?;
    let index = read(Some(index), &cupy, "index", "int", false)?;
    let data = read(Some(data), &cupy, "data", "float", false)?;
    let vals = read(Some(vals), &cupy, "vals", "float", false)?;
    launch(
        &cupy,
        &[&indptr, &index, &data, &vals],
        "domain_bbknn_cut_smaller",
        n_rows * 128,
        stream,
        vec![
            indptr.pointer(),
            index.pointer(),
            data.pointer(),
            vals.pointer(),
            n_rows,
        ],
    )?;
    Ok(())
}
#[pyfunction]
fn sort_tile_size() -> u64 {
    2048
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_bbknn_cuda")?;
    m.add_function(wrap_pyfunction!(find_top_k_per_row, &m)?)?;
    m.add_function(wrap_pyfunction!(find_top_k_per_row_sorted, &m)?)?;
    m.add_function(wrap_pyfunction!(sort_tile_size, &m)?)?;
    m.add_function(wrap_pyfunction!(cut_smaller, &m)?)?;
    Ok(())
}
