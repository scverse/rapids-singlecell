//! Native histogram kernels with explicit storage and launch validation.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Dtype, Layout},
    rank_support::*,
};
use pyo3::{exceptions::PyValueError, prelude::*};
#[pyfunction]
#[pyo3(signature=(X,gcodes,hist,*,n_cells,n_genes,n_groups,n_bins,bin_low,inv_bin_width,stream=0))]
pub fn dense_hist(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    gcodes: &Bound<'_, PyAny>,
    hist: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    n_groups: u64,
    n_bins: u64,
    bin_low: f64,
    inv_bin_width: f64,
    stream: usize,
) -> PyResult<()> {
    histogram_dense(
        &py.import("cupy")?,
        X,
        gcodes,
        hist,
        n_cells,
        n_genes,
        n_groups,
        n_bins,
        bin_low,
        inv_bin_width,
        false,
        stream,
    )
}
pub fn histogram_dense(
    cp: &Bound<'_, PyModule>,
    X: &Bound<'_, PyAny>,
    gcodes: &Bound<'_, PyAny>,
    hist: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    groups: u64,
    bins: u64,
    low: f64,
    inverse: f64,
    skip_zero: bool,
    stream: usize,
) -> PyResult<()> {
    if bins == 0 || bins > i32::MAX as u64 || !low.is_finite() || !inverse.is_finite() {
        return Err(PyValueError::new_err("invalid histogram bin configuration"));
    }
    let x = floating(X, cp, "X", Layout::F)?;
    capacity(&x, product(&[rows, cols])?, "X")?;
    let codes = vector(gcodes, cp, "gcodes", Some(Dtype::I32))?;
    capacity(&codes, rows, "gcodes")?;
    let h = read(hist, cp, "hist", Some(Dtype::U32), Layout::C)?;
    capacity(&h, product(&[cols, groups, bins + 1])?, "hist")?;
    h.require_disjoint(&x)?;
    h.require_disjoint(&codes)?;
    launch(
        cp,
        "rank_hist_dense",
        rows * cols,
        stream,
        &[&x, &codes, &h],
        &mut [
            Arg::P(x.pointer),
            Arg::P(codes.pointer),
            Arg::P(h.pointer),
            Arg::N(rows),
            Arg::N(cols),
            Arg::N(groups),
            Arg::N(bins),
            Arg::F(low),
            Arg::F(inverse),
            Arg::U(wide(x.dtype)?),
            Arg::U(skip_zero as u32),
        ],
    )
}
pub fn histogram_sparse(
    cp: &Bound<'_, PyModule>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    gcodes: &Bound<'_, PyAny>,
    hist: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    groups: u64,
    bins: u64,
    low: f64,
    inverse: f64,
    start: u64,
    csc: bool,
    stream: usize,
) -> PyResult<()> {
    if bins == 0 || bins > i32::MAX as u64 || !low.is_finite() || !inverse.is_finite() {
        return Err(PyValueError::new_err("invalid histogram bin configuration"));
    }
    let d = floating(data, cp, "data", Layout::C)?;
    d.require_vector("data")?;
    let ix = vector(indices, cp, "indices", None)?;
    let p = vector(indptr, cp, "indptr", Some(ix.dtype))?;
    if d.len != ix.len {
        return Err(PyValueError::new_err("data and indices lengths must match"));
    }
    let stop = start
        .checked_add(cols)
        .filter(|&stop| stop <= i64::MAX as u64)
        .ok_or_else(|| PyValueError::new_err("column range overflow"))?;
    let major = if csc { stop } else { rows };
    let pointer_count = major
        .checked_add(1)
        .ok_or_else(|| PyValueError::new_err("indptr length overflow"))?;
    capacity(&p, pointer_count, "indptr")?;
    let codes = vector(gcodes, cp, "gcodes", Some(Dtype::I32))?;
    capacity(&codes, rows, "gcodes")?;
    let h = read(hist, cp, "hist", Some(Dtype::U32), Layout::C)?;
    capacity(&h, product(&[cols, groups, bins + 1])?, "hist")?;
    for a in [&d, &ix, &p, &codes] {
        h.require_disjoint(a)?;
    }
    launch(
        cp,
        "rank_hist_sparse",
        product(&[if csc { cols } else { rows }, 32])?,
        stream,
        &[&d, &ix, &p, &codes, &h],
        &mut [
            Arg::P(d.pointer),
            Arg::P(ix.pointer),
            Arg::P(p.pointer),
            Arg::P(codes.pointer),
            Arg::P(h.pointer),
            Arg::N(rows),
            Arg::N(cols),
            Arg::N(groups),
            Arg::N(bins),
            Arg::F(low),
            Arg::F(inverse),
            Arg::N(start),
            Arg::N(d.len),
            Arg::U(wide(d.dtype)?),
            Arg::U(integer(&ix)?),
            Arg::U(csc as u32),
        ],
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,gcodes,hist,*,n_cells,n_genes,n_groups,n_bins,bin_low,inv_bin_width,gene_start,stream=0))]
pub fn csr_hist(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    gcodes: &Bound<'_, PyAny>,
    hist: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    n_groups: u64,
    n_bins: u64,
    bin_low: f64,
    inv_bin_width: f64,
    gene_start: u64,
    stream: usize,
) -> PyResult<()> {
    histogram_sparse(
        &py.import("cupy")?,
        data,
        indices,
        indptr,
        gcodes,
        hist,
        n_cells,
        n_genes,
        n_groups,
        n_bins,
        bin_low,
        inv_bin_width,
        gene_start,
        false,
        stream,
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,gcodes,hist,*,n_cells,n_genes,n_groups,n_bins,bin_low,inv_bin_width,gene_start,stream=0))]
pub fn csc_hist(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    gcodes: &Bound<'_, PyAny>,
    hist: &Bound<'_, PyAny>,
    n_cells: u64,
    n_genes: u64,
    n_groups: u64,
    n_bins: u64,
    bin_low: f64,
    inv_bin_width: f64,
    gene_start: u64,
    stream: usize,
) -> PyResult<()> {
    histogram_sparse(
        &py.import("cupy")?,
        data,
        indices,
        indptr,
        gcodes,
        hist,
        n_cells,
        n_genes,
        n_groups,
        n_bins,
        bin_low,
        inv_bin_width,
        gene_start,
        true,
        stream,
    )
}
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rapids_singlecell._cuda._wilcoxon_binned_cuda")?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(dense_hist, &m)?)?;
    m.add_function(wrap_pyfunction!(csr_hist, &m)?)?;
    m.add_function(wrap_pyfunction!(csc_hist, &m)?)?;
    parent.add_submodule(&m)
}
