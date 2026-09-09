//! autocorr bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(data_centered, *, adj_row_ptr, adj_col_ind, adj_data, num, n_samples, n_features, stream=0))]
fn morans_dense(
    py: Python<'_>,
    data_centered: &Bound<'_, PyAny>,
    adj_row_ptr: &Bound<'_, PyAny>,
    adj_col_ind: &Bound<'_, PyAny>,
    adj_data: &Bound<'_, PyAny>,
    num: &Bound<'_, PyAny>,
    n_samples: u64,
    n_features: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data_centered = read(Some(data_centered), &cupy, "data_centered", "T", false)?;
    let adj_row_ptr = read(Some(adj_row_ptr), &cupy, "adj_row_ptr", "Index", false)?;
    let adj_col_ind = read(Some(adj_col_ind), &cupy, "adj_col_ind", "Index", false)?;
    let adj_data = read(Some(adj_data), &cupy, "adj_data", "T", false)?;
    same(&adj_data, &data_centered)?;
    let num = read(Some(num), &cupy, "num", "T", false)?;
    same(&num, &data_centered)?;
    launch(
        &cupy,
        &[&data_centered, &adj_row_ptr, &adj_col_ind, &adj_data, &num],
        "domain_autocorr_morans_dense",
        n_samples * n_features,
        stream,
        vec![
            data_centered.pointer(),
            adj_row_ptr.pointer(),
            adj_col_ind.pointer(),
            adj_data.pointer(),
            num.pointer(),
            n_samples,
            n_features,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(adj_row_ptr, adj_col_ind, adj_data, *, data_row_ptr, data_col_ind, data_values, n_samples, n_features, mean_array, num, stream=0))]
fn morans_sparse(
    py: Python<'_>,
    adj_row_ptr: &Bound<'_, PyAny>,
    adj_col_ind: &Bound<'_, PyAny>,
    adj_data: &Bound<'_, PyAny>,
    data_row_ptr: &Bound<'_, PyAny>,
    data_col_ind: &Bound<'_, PyAny>,
    data_values: &Bound<'_, PyAny>,
    n_samples: u64,
    n_features: u64,
    mean_array: &Bound<'_, PyAny>,
    num: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let adj_row_ptr = read(Some(adj_row_ptr), &cupy, "adj_row_ptr", "AdjIdxT", false)?;
    let adj_col_ind = read(Some(adj_col_ind), &cupy, "adj_col_ind", "AdjIdxT", false)?;
    same(&adj_col_ind, &adj_row_ptr)?;
    let adj_data = read(Some(adj_data), &cupy, "adj_data", "T", false)?;
    let data_row_ptr = read(Some(data_row_ptr), &cupy, "data_row_ptr", "DataIdxT", false)?;
    let data_col_ind = read(Some(data_col_ind), &cupy, "data_col_ind", "DataIdxT", false)?;
    same(&data_col_ind, &data_row_ptr)?;
    let data_values = read(Some(data_values), &cupy, "data_values", "T", false)?;
    same(&data_values, &adj_data)?;
    let mean_array = read(Some(mean_array), &cupy, "mean_array", "T", false)?;
    same(&mean_array, &adj_data)?;
    let num = read(Some(num), &cupy, "num", "T", false)?;
    same(&num, &adj_data)?;
    launch(
        &cupy,
        &[
            &adj_row_ptr,
            &adj_col_ind,
            &adj_data,
            &data_row_ptr,
            &data_col_ind,
            &data_values,
            &mean_array,
            &num,
        ],
        "domain_autocorr_morans_sparse",
        n_samples * 128,
        stream,
        vec![
            adj_row_ptr.pointer(),
            adj_col_ind.pointer(),
            adj_data.pointer(),
            data_row_ptr.pointer(),
            data_col_ind.pointer(),
            data_values.pointer(),
            n_samples,
            n_features,
            mean_array.pointer(),
            num.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(data, *, adj_row_ptr, adj_col_ind, adj_data, num, n_samples, n_features, stream=0))]
fn gearys_dense(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    adj_row_ptr: &Bound<'_, PyAny>,
    adj_col_ind: &Bound<'_, PyAny>,
    adj_data: &Bound<'_, PyAny>,
    num: &Bound<'_, PyAny>,
    n_samples: u64,
    n_features: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "T", false)?;
    let adj_row_ptr = read(Some(adj_row_ptr), &cupy, "adj_row_ptr", "Index", false)?;
    let adj_col_ind = read(Some(adj_col_ind), &cupy, "adj_col_ind", "Index", false)?;
    let adj_data = read(Some(adj_data), &cupy, "adj_data", "T", false)?;
    same(&adj_data, &data)?;
    let num = read(Some(num), &cupy, "num", "T", false)?;
    same(&num, &data)?;
    launch(
        &cupy,
        &[&data, &adj_row_ptr, &adj_col_ind, &adj_data, &num],
        "domain_autocorr_gearys_dense",
        n_samples * n_features,
        stream,
        vec![
            data.pointer(),
            adj_row_ptr.pointer(),
            adj_col_ind.pointer(),
            adj_data.pointer(),
            num.pointer(),
            n_samples,
            n_features,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(adj_row_ptr, adj_col_ind, adj_data, *, data_row_ptr, data_col_ind, data_values, n_samples, n_features, num, stream=0))]
fn gearys_sparse(
    py: Python<'_>,
    adj_row_ptr: &Bound<'_, PyAny>,
    adj_col_ind: &Bound<'_, PyAny>,
    adj_data: &Bound<'_, PyAny>,
    data_row_ptr: &Bound<'_, PyAny>,
    data_col_ind: &Bound<'_, PyAny>,
    data_values: &Bound<'_, PyAny>,
    n_samples: u64,
    n_features: u64,
    num: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let adj_row_ptr = read(Some(adj_row_ptr), &cupy, "adj_row_ptr", "AdjIdxT", false)?;
    let adj_col_ind = read(Some(adj_col_ind), &cupy, "adj_col_ind", "AdjIdxT", false)?;
    same(&adj_col_ind, &adj_row_ptr)?;
    let adj_data = read(Some(adj_data), &cupy, "adj_data", "T", false)?;
    let data_row_ptr = read(Some(data_row_ptr), &cupy, "data_row_ptr", "DataIdxT", false)?;
    let data_col_ind = read(Some(data_col_ind), &cupy, "data_col_ind", "DataIdxT", false)?;
    same(&data_col_ind, &data_row_ptr)?;
    let data_values = read(Some(data_values), &cupy, "data_values", "T", false)?;
    same(&data_values, &adj_data)?;
    let num = read(Some(num), &cupy, "num", "T", false)?;
    same(&num, &adj_data)?;
    launch(
        &cupy,
        &[
            &adj_row_ptr,
            &adj_col_ind,
            &adj_data,
            &data_row_ptr,
            &data_col_ind,
            &data_values,
            &num,
        ],
        "domain_autocorr_gearys_sparse",
        n_samples * 128,
        stream,
        vec![
            adj_row_ptr.pointer(),
            adj_col_ind.pointer(),
            adj_data.pointer(),
            data_row_ptr.pointer(),
            data_col_ind.pointer(),
            data_values.pointer(),
            n_samples,
            n_features,
            num.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(data_col_ind, data_values, *, nnz, mean_array, den, counter, stream=0))]
fn pre_den_sparse(
    py: Python<'_>,
    data_col_ind: &Bound<'_, PyAny>,
    data_values: &Bound<'_, PyAny>,
    nnz: u64,
    mean_array: &Bound<'_, PyAny>,
    den: &Bound<'_, PyAny>,
    counter: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data_col_ind = read(Some(data_col_ind), &cupy, "data_col_ind", "IdxT", false)?;
    let data_values = read(Some(data_values), &cupy, "data_values", "T", false)?;
    let mean_array = read(Some(mean_array), &cupy, "mean_array", "T", false)?;
    same(&mean_array, &data_values)?;
    let den = read(Some(den), &cupy, "den", "T", false)?;
    same(&den, &data_values)?;
    let counter = read(Some(counter), &cupy, "counter", "int", false)?;
    launch(
        &cupy,
        &[&data_col_ind, &data_values, &mean_array, &den, &counter],
        "domain_autocorr_pre_den_sparse",
        nnz,
        stream,
        vec![
            data_col_ind.pointer(),
            data_values.pointer(),
            nnz,
            mean_array.pointer(),
            den.pointer(),
            counter.pointer(),
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_autocorr_cuda")?;
    m.add_function(wrap_pyfunction!(morans_dense, &m)?)?;
    m.add_function(wrap_pyfunction!(morans_sparse, &m)?)?;
    m.add_function(wrap_pyfunction!(gearys_dense, &m)?)?;
    m.add_function(wrap_pyfunction!(gearys_sparse, &m)?)?;
    m.add_function(wrap_pyfunction!(pre_den_sparse, &m)?)?;
    Ok(())
}
