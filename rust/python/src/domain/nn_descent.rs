//! nn_descent bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(data, *, out, pairs, n_samples, n_features, n_neighbors, stream=0))]
fn sqeuclidean(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    pairs: &Bound<'_, PyAny>,
    n_samples: u64,
    n_features: u64,
    n_neighbors: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "float", false)?;
    let out = read(Some(out), &cupy, "out", "float", false)?;
    let pairs = read(Some(pairs), &cupy, "pairs", "unsigned int", false)?;
    launch(
        &cupy,
        &[&data, &out, &pairs],
        "domain_nn_descent_sqeuclidean",
        n_samples * n_neighbors,
        stream,
        vec![
            data.pointer(),
            out.pointer(),
            pairs.pointer(),
            n_samples,
            n_features,
            n_neighbors,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(data, *, out, pairs, n_samples, n_features, n_neighbors, stream=0))]
fn cosine(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    pairs: &Bound<'_, PyAny>,
    n_samples: u64,
    n_features: u64,
    n_neighbors: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "float", false)?;
    let out = read(Some(out), &cupy, "out", "float", false)?;
    let pairs = read(Some(pairs), &cupy, "pairs", "unsigned int", false)?;
    launch(
        &cupy,
        &[&data, &out, &pairs],
        "domain_nn_descent_cosine",
        n_samples * n_neighbors,
        stream,
        vec![
            data.pointer(),
            out.pointer(),
            pairs.pointer(),
            n_samples,
            n_features,
            n_neighbors,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(data, *, out, pairs, n_samples, n_features, n_neighbors, stream=0))]
fn inner(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    pairs: &Bound<'_, PyAny>,
    n_samples: u64,
    n_features: u64,
    n_neighbors: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "float", false)?;
    let out = read(Some(out), &cupy, "out", "float", false)?;
    let pairs = read(Some(pairs), &cupy, "pairs", "unsigned int", false)?;
    launch(
        &cupy,
        &[&data, &out, &pairs],
        "domain_nn_descent_inner",
        n_samples * n_neighbors,
        stream,
        vec![
            data.pointer(),
            out.pointer(),
            pairs.pointer(),
            n_samples,
            n_features,
            n_neighbors,
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_nn_descent_cuda")?;
    m.add_function(wrap_pyfunction!(sqeuclidean, &m)?)?;
    m.add_function(wrap_pyfunction!(cosine, &m)?)?;
    m.add_function(wrap_pyfunction!(inner, &m)?)?;
    Ok(())
}
