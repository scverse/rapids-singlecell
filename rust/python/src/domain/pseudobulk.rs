//! pseudobulk bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(X, Y, *, out, n_pairs, n_features, stream=0))]
fn paired_squared(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    Y: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    n_pairs: u64,
    n_features: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let X = read(Some(X), &cupy, "X", "double", false)?;
    let Y = read(Some(Y), &cupy, "Y", "double", false)?;
    let out = read(Some(out), &cupy, "out", "double", false)?;
    launch(
        &cupy,
        &[&X, &Y, &out],
        "domain_pseudobulk_paired_squared",
        n_pairs * 128,
        stream,
        vec![X.pointer(), Y.pointer(), out.pointer(), n_pairs, n_features],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X, Y, *, out, n_pairs, n_features, stream=0))]
fn paired_abs_mean(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    Y: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    n_pairs: u64,
    n_features: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let X = read(Some(X), &cupy, "X", "double", false)?;
    let Y = read(Some(Y), &cupy, "Y", "double", false)?;
    let out = read(Some(out), &cupy, "out", "double", false)?;
    launch(
        &cupy,
        &[&X, &Y, &out],
        "domain_pseudobulk_paired_abs_mean",
        n_pairs * 128,
        stream,
        vec![X.pointer(), Y.pointer(), out.pointer(), n_pairs, n_features],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X, Y, *, out, n_x, n_y, n_features, stream=0))]
fn pairwise_squared(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    Y: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    n_x: u64,
    n_y: u64,
    n_features: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let X = read(Some(X), &cupy, "X", "double", false)?;
    let Y = read(Some(Y), &cupy, "Y", "double", false)?;
    let out = read(Some(out), &cupy, "out", "double", false)?;
    launch(
        &cupy,
        &[&X, &Y, &out],
        "domain_pseudobulk_pairwise_squared",
        (n_x * n_y) * 128,
        stream,
        vec![
            X.pointer(),
            Y.pointer(),
            out.pointer(),
            n_x,
            n_y,
            n_features,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(X, Y, *, out, n_x, n_y, n_features, stream=0))]
fn pairwise_abs_mean(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    Y: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    n_x: u64,
    n_y: u64,
    n_features: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let X = read(Some(X), &cupy, "X", "double", false)?;
    let Y = read(Some(Y), &cupy, "Y", "double", false)?;
    let out = read(Some(out), &cupy, "out", "double", false)?;
    launch(
        &cupy,
        &[&X, &Y, &out],
        "domain_pseudobulk_pairwise_abs_mean",
        (n_x * n_y) * 128,
        stream,
        vec![
            X.pointer(),
            Y.pointer(),
            out.pointer(),
            n_x,
            n_y,
            n_features,
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_pseudobulk_cuda")?;
    m.add_function(wrap_pyfunction!(paired_squared, &m)?)?;
    m.add_function(wrap_pyfunction!(paired_abs_mean, &m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_squared, &m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_abs_mean, &m)?)?;
    Ok(())
}
