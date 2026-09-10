//! spca bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(indptr, index, data, *, nrows, ncols, out, stream=0))]
fn gram_csr_upper(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    nrows: u64,
    ncols: u64,
    out: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let indptr = read(Some(indptr), &cupy, "indptr", "IdxT", false)?;
    let index = read(Some(index), &cupy, "index", "IdxT", false)?;
    same(&index, &indptr)?;
    let data = read(Some(data), &cupy, "data", "T", false)?;
    let out = read(Some(out), &cupy, "out", "T", false)?;
    same(&out, &data)?;
    let matrix_size = ncols
        .checked_mul(ncols)
        .ok_or_else(|| PyValueError::new_err("Gram matrix dimensions overflow"))?;
    if nrows >= indptr.len() || matrix_size > out.len() {
        return Err(PyValueError::new_err(
            "Gram matrix dimensions exceed an input or output allocation",
        ));
    }

    launch(
        &cupy,
        &[&indptr, &index, &data, &out],
        "domain_spca_gram_csr_upper",
        nrows * 128,
        stream,
        vec![
            indptr.pointer(),
            index.pointer(),
            data.pointer(),
            nrows,
            ncols,
            out.pointer(),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(*, out, ncols, stream=0))]
fn copy_upper_to_lower(
    py: Python<'_>,
    out: &Bound<'_, PyAny>,
    ncols: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let out = read(Some(out), &cupy, "out", "T", false)?;
    launch(
        &cupy,
        &[&out],
        "domain_spca_copy_upper_to_lower",
        ncols * ncols,
        stream,
        vec![out.pointer(), ncols],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(gram, meanx, meany, *, cov, ncols, stream=0))]
fn cov_from_gram(
    py: Python<'_>,
    gram: &Bound<'_, PyAny>,
    meanx: &Bound<'_, PyAny>,
    meany: &Bound<'_, PyAny>,
    cov: &Bound<'_, PyAny>,
    ncols: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let gram = read(Some(gram), &cupy, "gram", "T", false)?;
    let meanx = read(Some(meanx), &cupy, "meanx", "T", false)?;
    same(&meanx, &gram)?;
    let meany = read(Some(meany), &cupy, "meany", "T", false)?;
    same(&meany, &gram)?;
    let cov = read(Some(cov), &cupy, "cov", "T", false)?;
    same(&cov, &gram)?;
    launch(
        &cupy,
        &[&gram, &meanx, &meany, &cov],
        "domain_spca_cov_from_gram",
        ncols * ncols,
        stream,
        vec![
            gram.pointer(),
            meanx.pointer(),
            meany.pointer(),
            cov.pointer(),
            ncols,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(indices, *, out, nnz, num_genes, stream=0))]
fn check_zero_genes(
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    nnz: u64,
    num_genes: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let indices = read(Some(indices), &cupy, "indices", "IdxT", false)?;
    let out = read(Some(out), &cupy, "out", "int", false)?;
    launch(
        &cupy,
        &[&indices, &out],
        "domain_spca_check_zero_genes",
        nnz,
        stream,
        vec![indices.pointer(), out.pointer(), nnz, num_genes],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_spca_cuda")?;
    m.add_function(wrap_pyfunction!(gram_csr_upper, &m)?)?;
    m.add_function(wrap_pyfunction!(copy_upper_to_lower, &m)?)?;
    m.add_function(wrap_pyfunction!(cov_from_gram, &m)?)?;
    m.add_function(wrap_pyfunction!(check_zero_genes, &m)?)?;
    Ok(())
}
