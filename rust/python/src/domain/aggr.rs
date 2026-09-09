//! aggr bindings. All CUDA allocations are borrowed from CuPy.
#![allow(non_snake_case, clippy::too_many_arguments)]
use super::*;
#[pyfunction]
#[pyo3(signature=(indptr, index, data, *, out_sum=None, out_count=None, out_sqsum=None, cats, mask, n_cells, n_genes, is_csc, stream=0))]
fn sparse_aggr(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    out_sum: Option<&Bound<'_, PyAny>>,
    out_count: Option<&Bound<'_, PyAny>>,
    out_sqsum: Option<&Bound<'_, PyAny>>,
    cats: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    is_csc: bool,
    stream: usize,
) -> PyResult<()> {
    if out_sum.is_none() && out_count.is_none() && out_sqsum.is_none() {
        return Err(PyValueError::new_err(
            "at least one aggregate output is required",
        ));
    }
    let cupy = py.import("cupy")?;

    let indptr = read(Some(indptr), &cupy, "indptr", "IdxT", false)?;
    let index = read(Some(index), &cupy, "index", "IdxT", false)?;
    same(&index, &indptr)?;
    let data = read(Some(data), &cupy, "data", "T", false)?;
    let out_sum = read(out_sum, &cupy, "out_sum", "double", false)?;
    let out_count = read(out_count, &cupy, "out_count", "double", false)?;
    let out_sqsum = read(out_sqsum, &cupy, "out_sqsum", "double", false)?;
    let cats = read(Some(cats), &cupy, "cats", "int", false)?;
    let mask = read(Some(mask), &cupy, "mask", "bool", false)?;
    launch(
        &cupy,
        &[
            &indptr, &index, &data, &out_sum, &out_count, &out_sqsum, &cats, &mask,
        ],
        "domain_aggr_sparse_aggr",
        (if is_csc { n_genes } else { n_cells }) * 128,
        stream,
        vec![
            indptr.pointer(),
            index.pointer(),
            data.pointer(),
            out_sum.pointer(),
            out_count.pointer(),
            out_sqsum.pointer(),
            cats.pointer(),
            mask.pointer(),
            n_cells,
            n_genes,
            u64::from(is_csc),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(data, *, out_sum=None, out_count=None, out_sqsum=None, cats, mask, n_cells, n_genes, is_fortran, stream=0))]
fn dense_aggr(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    out_sum: Option<&Bound<'_, PyAny>>,
    out_count: Option<&Bound<'_, PyAny>>,
    out_sqsum: Option<&Bound<'_, PyAny>>,
    cats: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    is_fortran: bool,
    stream: usize,
) -> PyResult<()> {
    if out_sum.is_none() && out_count.is_none() && out_sqsum.is_none() {
        return Err(PyValueError::new_err(
            "at least one aggregate output is required",
        ));
    }
    let cupy = py.import("cupy")?;

    let data = read(Some(data), &cupy, "data", "T", true)?;
    let out_sum = read(out_sum, &cupy, "out_sum", "double", false)?;
    let out_count = read(out_count, &cupy, "out_count", "double", false)?;
    let out_sqsum = read(out_sqsum, &cupy, "out_sqsum", "double", false)?;
    let cats = read(Some(cats), &cupy, "cats", "int", false)?;
    let mask = read(Some(mask), &cupy, "mask", "bool", false)?;
    launch(
        &cupy,
        &[&data, &out_sum, &out_count, &out_sqsum, &cats, &mask],
        "domain_aggr_dense_aggr",
        n_cells * n_genes,
        stream,
        vec![
            data.pointer(),
            out_sum.pointer(),
            out_count.pointer(),
            out_sqsum.pointer(),
            cats.pointer(),
            mask.pointer(),
            n_cells,
            n_genes,
            u64::from(is_fortran),
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(indptr, index, data, *, out_row, out_col, out_data, cats, mask, n_cells, stream=0))]
fn csr_to_coo(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    out_row: &Bound<'_, PyAny>,
    out_col: &Bound<'_, PyAny>,
    out_data: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    n_cells: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let indptr = read(Some(indptr), &cupy, "indptr", "IdxT", false)?;
    let index = read(Some(index), &cupy, "index", "IdxT", false)?;
    same(&index, &indptr)?;
    let data = read(Some(data), &cupy, "data", "T", false)?;
    let out_row = read(Some(out_row), &cupy, "out_row", "OutIdxT", false)?;
    same(&out_row, &indptr)?;
    let out_col = read(Some(out_col), &cupy, "out_col", "OutIdxT", false)?;
    same(&out_col, &indptr)?;
    let out_data = read(Some(out_data), &cupy, "out_data", "double", false)?;
    let cats = read(Some(cats), &cupy, "cats", "int", false)?;
    let mask = read(Some(mask), &cupy, "mask", "bool", false)?;
    launch(
        &cupy,
        &[
            &indptr, &index, &data, &out_row, &out_col, &out_data, &cats, &mask,
        ],
        "domain_aggr_csr_to_coo",
        n_cells * 128,
        stream,
        vec![
            indptr.pointer(),
            index.pointer(),
            data.pointer(),
            out_row.pointer(),
            out_col.pointer(),
            out_data.pointer(),
            cats.pointer(),
            mask.pointer(),
            n_cells,
        ],
    )?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(indptr, index, data, *, means, n_cells, dof, n_groups, stream=0))]
fn sparse_var(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    index: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    means: &Bound<'_, PyAny>,
    n_cells: &Bound<'_, PyAny>,
    dof: u64,
    n_groups: u64,
    stream: usize,
) -> PyResult<()> {
    let cupy = py.import("cupy")?;

    let indptr = read(Some(indptr), &cupy, "indptr", "IdxT", false)?;
    let index = read(Some(index), &cupy, "index", "IdxT", false)?;
    same(&index, &indptr)?;
    let data = read(Some(data), &cupy, "data", "double", false)?;
    let means = read(Some(means), &cupy, "means", "double", false)?;
    let n_cells = read(Some(n_cells), &cupy, "n_cells", "double", false)?;
    launch(
        &cupy,
        &[&indptr, &index, &data, &means, &n_cells],
        "domain_aggr_sparse_var",
        n_groups * 128,
        stream,
        vec![
            indptr.pointer(),
            index.pointer(),
            data.pointer(),
            means.pointer(),
            n_cells.pointer(),
            dof,
            n_groups,
        ],
    )?;
    Ok(())
}
pub(super) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = submodule(parent, "_aggr_cuda")?;
    m.add_function(wrap_pyfunction!(sparse_aggr, &m)?)?;
    m.add_function(wrap_pyfunction!(dense_aggr, &m)?)?;
    m.add_function(wrap_pyfunction!(csr_to_coo, &m)?)?;
    m.add_function(wrap_pyfunction!(sparse_var, &m)?)?;
    Ok(())
}
