//! Sparse extraction and multiple-testing/group-statistics bindings.
#![allow(clippy::too_many_arguments)]
use crate::{
    array::{Dtype, Layout},
    rank_support::*,
};
use pyo3::{exceptions::PyValueError, prelude::*};
#[pyfunction]
#[pyo3(signature=(values,*,stream=0))]
pub fn fdr_bh_reverse_cummin(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let a = read(values, &cp, "values", Some(Dtype::F64), Layout::C)?;
    let (rows, cols) = matrix(&a, "values")?;
    launch(
        &cp,
        "rank_bh",
        rows,
        stream,
        &[&a],
        &mut [Arg::P(a.pointer), Arg::N(rows), Arg::N(cols)],
    )
}
#[allow(clippy::too_many_arguments)]
pub fn tile(
    cp: &Bound<'_, PyModule>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    lb: isize,
    ub: isize,
    csc: bool,
    stream: usize,
) -> PyResult<()> {
    if lb < 0 || ub < lb {
        return Err(PyValueError::new_err("invalid sparse column window"));
    }
    let p = vector(indptr, cp, "indptr", None)?;
    let ix = vector(indices, cp, "indices", None)?;
    let d = floating(data, cp, "data", Layout::C)?;
    d.require_vector("data")?;
    let o = read(out, cp, "out", Some(Dtype::F64), Layout::F)?;
    let (rows, cols) = matrix(&o, "out")?;
    if p.len == 0
        || ix.len != d.len
        || cols < (ub - lb) as u64
        || (!csc && p.len != rows + 1)
        || (csc && p.len <= ub as u64)
    {
        return Err(PyValueError::new_err(
            "inconsistent sparse input or dense output shape",
        ));
    }
    for a in [&p, &ix, &d] {
        o.require_disjoint(a)?;
    }
    launch(
        cp,
        "rank_tile",
        if csc {
            product(&[(ub - lb) as u64, 128])?
        } else {
            rows
        },
        stream,
        &[&p, &ix, &d, &o],
        &mut [
            Arg::P(p.pointer),
            Arg::P(ix.pointer),
            Arg::P(d.pointer),
            Arg::P(o.pointer),
            Arg::N(rows),
            Arg::N((ub - lb) as u64),
            Arg::N(lb as u64),
            Arg::N(p.len - 1),
            Arg::N(d.len),
            Arg::U(integer(&p)?),
            Arg::U(integer(&ix)?),
            Arg::U(wide(d.dtype)?),
            Arg::U(csc as u32),
        ],
    )
}
#[pyfunction]
#[pyo3(signature=(indptr,indices,data,out,*,col_lb,col_ub,stream=0))]
pub fn csr_tile_to_dense(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    col_lb: isize,
    col_ub: isize,
    stream: usize,
) -> PyResult<()> {
    tile(
        &py.import("cupy")?,
        indptr,
        indices,
        data,
        out,
        col_lb,
        col_ub,
        false,
        stream,
    )
}
#[pyfunction]
#[pyo3(signature=(indptr,indices,data,out,*,col_lb,col_ub,stream=0))]
pub fn csc_tile_to_dense(
    py: Python<'_>,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    col_lb: isize,
    col_ub: isize,
    stream: usize,
) -> PyResult<()> {
    tile(
        &py.import("cupy")?,
        indptr,
        indices,
        data,
        out,
        col_lb,
        col_ub,
        true,
        stream,
    )
}
#[pyfunction]
#[pyo3(signature=(block,group_codes,group_sums,group_sum_sq,group_nnz,*,compute_nnz,stream=0))]
pub fn group_chunk_stats(
    py: Python<'_>,
    block: &Bound<'_, PyAny>,
    group_codes: &Bound<'_, PyAny>,
    group_sums: &Bound<'_, PyAny>,
    group_sum_sq: &Bound<'_, PyAny>,
    group_nnz: &Bound<'_, PyAny>,
    compute_nnz: bool,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let b = read(block, &cp, "block", Some(Dtype::F64), Layout::F)?;
    let (rows, cols) = matrix(&b, "block")?;
    let sums = read(group_sums, &cp, "group_sums", Some(Dtype::F64), Layout::C)?;
    let (groups, output_cols) = matrix(&sums, "group_sums")?;
    let squares = read(
        group_sum_sq,
        &cp,
        "group_sum_sq",
        Some(Dtype::F64),
        Layout::C,
    )?;
    let codes = vector(group_codes, &cp, "group_codes", Some(Dtype::I32))?;
    if output_cols != cols
        || matrix(&squares, "group_sum_sq")? != (groups, cols)
        || codes.len != rows
    {
        return Err(PyValueError::new_err(
            "group statistics input and output shapes must match",
        ));
    }
    if compute_nnz {
        let counts = read(group_nnz, &cp, "group_nnz", Some(Dtype::F64), Layout::C)?;
        if matrix(&counts, "group_nnz")? != (groups, cols) {
            return Err(PyValueError::new_err(
                "group_nnz must have shape (n_groups, n_cols)",
            ));
        }
    }
    stats(
        &cp,
        block,
        group_codes,
        None,
        Some(group_sums),
        Some(group_sum_sq),
        if compute_nnz { Some(group_nnz) } else { None },
        None,
        None,
        cols,
        0,
        stream,
        false,
    )
}
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rapids_singlecell._cuda._rank_stats_cuda")?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(fdr_bh_reverse_cummin, &m)?)?;
    m.add_function(wrap_pyfunction!(csr_tile_to_dense, &m)?)?;
    m.add_function(wrap_pyfunction!(csc_tile_to_dense, &m)?)?;
    m.add_function(wrap_pyfunction!(group_chunk_stats, &m)?)?;
    parent.add_submodule(&m)
}
