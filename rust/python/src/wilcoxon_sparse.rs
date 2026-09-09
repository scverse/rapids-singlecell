//! Sparse exact ranking, bounded column staging, and CSR range utilities.
#![allow(clippy::too_many_arguments)]
use crate::sparse_ovr::{self, CscBlock};
use crate::{
    array::{Dtype, Layout, current_device},
    rank_support::*,
    runtime::StreamScope,
    sparse_ovo,
    wilcoxon::{self, Stats},
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

fn csc_block<'py>(
    cp: &Bound<'py, PyModule>,
    source: &Bound<'py, PyAny>,
    host: bool,
) -> PyResult<CscBlock<'py>> {
    let source = source.call_method0("tocsc")?;
    let data = source.getattr("data")?;
    let indices = source.getattr("indices")?;
    let indptr = source.getattr("indptr")?;
    if !host {
        return Ok(CscBlock {
            data,
            indices,
            indptr,
        });
    }
    let dtype: String = data.getattr("dtype")?.getattr("name")?.extract()?;
    let idtype: String = indices.getattr("dtype")?.getattr("name")?.extract()?;
    let pdtype: String = indptr.getattr("dtype")?.getattr("name")?.extract()?;
    Ok(CscBlock {
        data: array(cp, &data, &dtype, "C")?,
        indices: array(cp, &indices, &idtype, "C")?,
        indptr: array(cp, &indptr, &pdtype, "C")?,
    })
}

fn host_vector(object: &Bound<'_, PyAny>, name: &str, integer_only: bool) -> PyResult<()> {
    let np = object.py().import("numpy")?;
    if !object.is_instance(&np.getattr("ndarray")?)?
        || object.getattr("ndim")?.extract::<usize>()? != 1
    {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a one-dimensional NumPy array"
        )));
    }
    if !object
        .getattr("flags")?
        .getattr("c_contiguous")?
        .extract::<bool>()?
    {
        return Err(PyValueError::new_err(format!("{name} must be contiguous")));
    }
    let d: String = object.getattr("dtype")?.getattr("name")?.extract()?;
    if if integer_only {
        d != "int32" && d != "int64"
    } else {
        d != "float32" && d != "float64"
    } {
        return Err(PyTypeError::new_err(format!("unsupported {name} dtype")));
    }
    Ok(())
}
/// Host sparse objects borrow the original NumPy storage. Column slicing compacts
/// only the current window before any data are copied to the GPU.
fn host_sparse<'py>(
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
    csc: bool,
) -> PyResult<Bound<'py, PyAny>> {
    host_vector(data, "data", false)?;
    host_vector(indices, "indices", true)?;
    host_vector(indptr, "indptr", true)?;
    if data.len()? != indices.len()? || indptr.len()? != if csc { cols + 1 } else { rows + 1 } {
        return Err(PyValueError::new_err("inconsistent sparse array lengths"));
    }
    let kw = PyDict::new(data.py());
    kw.set_item("shape", (rows, cols))?;
    kw.set_item("copy", false)?;
    let matrix = data
        .py()
        .import("scipy.sparse")?
        .getattr(if csc { "csc_matrix" } else { "csr_matrix" })?
        .call(((data, indices, indptr),), Some(&kw))?;
    // Checking offsets before native sparse slicing prevents malformed metadata
    // from escaping through the host library's unchecked indexing paths.
    matrix.call_method1("check_format", (true,))?;
    Ok(matrix)
}
fn validate_device_sparse(
    cp: &Bound<'_, PyModule>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    rows: usize,
    cols: usize,
    csc: bool,
    ranks: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let d = floating(data, cp, "data", Layout::C)?;
    d.require_vector("data")?;
    let i = vector(indices, cp, "indices", None)?;
    integer(&i)?;
    let p = vector(indptr, cp, "indptr", None)?;
    integer(&p)?;
    if d.len != i.len || p.len != if csc { cols + 1 } else { rows + 1 } as u64 {
        return Err(PyValueError::new_err("inconsistent sparse input lengths"));
    }
    let r = read(ranks, cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    for a in [&d, &i, &p] {
        r.require_disjoint(a)?;
    }
    current_device(cp, &[&d, &i, &p, &r])?;
    Ok(())
}
fn ovr_device(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    codes: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    ranks: &Bound<'_, PyAny>,
    tie: &Bound<'_, PyAny>,
    compute: bool,
    batch: isize,
    csc: bool,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let c = vector(codes, &cp, "group_codes", Some(Dtype::I32))?;
    let sz = vector(sizes, &cp, "group_sizes", Some(Dtype::F64))?;
    let out = shape(ranks)?;
    if out.len() != 2 || out[0] != sz.len as usize {
        return Err(PyValueError::new_err("rank_sums must have n_groups rows"));
    }
    validate_device_sparse(
        &cp,
        data,
        indices,
        indptr,
        c.len as usize,
        out[1],
        csc,
        ranks,
    )?;
    if compute {
        let t = vector(tie, &cp, "tie_corr", Some(Dtype::F64))?;
        if t.len != out[1] as u64 {
            return Err(PyValueError::new_err(
                "tie_corr must match the output columns",
            ));
        }
        for obj in [data, indices, indptr] {
            t.require_disjoint(&read(obj, &cp, "sparse input", None, Layout::C)?)?;
        }
    }
    let source = if csc {
        None
    } else {
        let kwargs = PyDict::new(py);
        kwargs.set_item("shape", (c.len as usize, out[1]))?;
        kwargs.set_item("copy", false)?;
        Some(
            py.import("cupyx.scipy.sparse")?
                .getattr("csr_matrix")?
                .call(((data, indices, indptr),), Some(&kwargs))?,
        )
    };
    sparse_ovr::ovr(
        &cp,
        c.len as usize,
        codes,
        sizes,
        ranks,
        tie,
        compute,
        batch,
        None,
        |a, b| {
            if let Some(source) = &source {
                csc_block(&cp, &window(source, a, b)?, false)
            } else {
                Ok(CscBlock {
                    data: data.clone(),
                    indices: indices.clone(),
                    indptr: indptr.get_item(slice(py, a, b + 1))?,
                })
            }
        },
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,group_codes,group_sizes,rank_sums,tie_corr,*,compute_tie_corr,sub_batch_cols=64))]
pub fn ovr_sparse_csr_device(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    group_codes: &Bound<'_, PyAny>,
    group_sizes: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
) -> PyResult<()> {
    ovr_device(
        py,
        data,
        indices,
        indptr,
        group_codes,
        group_sizes,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        false,
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,group_codes,group_sizes,rank_sums,tie_corr,*,compute_tie_corr,sub_batch_cols=64))]
pub fn ovr_sparse_csc_device(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    group_codes: &Bound<'_, PyAny>,
    group_sizes: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
) -> PyResult<()> {
    ovr_device(
        py,
        data,
        indices,
        indptr,
        group_codes,
        group_sizes,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        true,
    )
}

fn ovr_host(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    codes: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    ranks: &Bound<'_, PyAny>,
    tie: &Bound<'_, PyAny>,
    sums: &Bound<'_, PyAny>,
    nnz: &Bound<'_, PyAny>,
    total: &Bound<'_, PyAny>,
    total_nnz: &Bound<'_, PyAny>,
    n_cols: usize,
    compute: bool,
    compute_nnz: bool,
    compute_totals: bool,
    start: isize,
    stop: isize,
    batch: isize,
    csc: bool,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    host_vector(codes, "group_codes", true)?;
    let rows = codes.len()?;
    let (start, stop) = range(start, stop, n_cols)?;
    let out = shape(ranks)?;
    if out != [sizes.len()?, stop - start] {
        return Err(PyValueError::new_err(
            "rank_sums must have shape (n_groups, window_cols)",
        ));
    }
    let source = host_sparse(data, indices, indptr, rows, n_cols, csc)?;
    let gc = array(&cp, codes, "int32", "C")?;
    let gs = array(&cp, sizes, "float64", "C")?;
    let st = Stats {
        sums,
        nnz: compute_nnz.then_some(nnz),
        total: compute_totals.then_some(total),
        total_nnz: (compute_nnz && compute_totals).then_some(total_nnz),
    };
    sparse_ovr::ovr(
        &cp,
        rows,
        &gc,
        &gs,
        ranks,
        tie,
        compute,
        batch,
        Some(st),
        |a, b| csc_block(&cp, &window(&source, start + a, start + b)?, true),
    )
}
#[pyfunction]
#[pyo3(signature=(h_data,h_indices,h_indptr,h_group_codes,h_group_sizes,d_rank_sums,d_tie_corr,d_group_sums,d_group_nnz,d_total_sums,d_total_nnz,*,compute_tie_corr,compute_nnz=true,compute_totals=false,sub_batch_cols=64))]
pub fn ovr_sparse_csc_host(
    py: Python<'_>,
    h_data: &Bound<'_, PyAny>,
    h_indices: &Bound<'_, PyAny>,
    h_indptr: &Bound<'_, PyAny>,
    h_group_codes: &Bound<'_, PyAny>,
    h_group_sizes: &Bound<'_, PyAny>,
    d_rank_sums: &Bound<'_, PyAny>,
    d_tie_corr: &Bound<'_, PyAny>,
    d_group_sums: &Bound<'_, PyAny>,
    d_group_nnz: &Bound<'_, PyAny>,
    d_total_sums: &Bound<'_, PyAny>,
    d_total_nnz: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    sub_batch_cols: isize,
) -> PyResult<()> {
    let cols = h_indptr
        .len()?
        .checked_sub(1)
        .ok_or_else(|| PyValueError::new_err("indptr must not be empty"))?;
    ovr_host(
        py,
        h_data,
        h_indices,
        h_indptr,
        h_group_codes,
        h_group_sizes,
        d_rank_sums,
        d_tie_corr,
        d_group_sums,
        d_group_nnz,
        d_total_sums,
        d_total_nnz,
        cols,
        compute_tie_corr,
        compute_nnz,
        compute_totals,
        0,
        -1,
        sub_batch_cols,
        true,
    )
}
#[pyfunction]
#[pyo3(signature=(h_data,h_indices,h_indptr,h_row_starts,h_row_stops,h_group_codes,h_group_sizes,d_rank_sums,d_tie_corr,d_group_sums,d_group_nnz,d_total_sums,d_total_nnz,*,n_cols,compute_tie_corr,compute_nnz=true,compute_totals=false,col_start=0,col_stop=-1,sub_batch_cols=64))]
#[pyo3(
    text_signature = "(h_data, h_indices, h_indptr, h_row_starts, h_row_stops, h_group_codes, h_group_sizes, d_rank_sums, d_tie_corr, d_group_sums, d_group_nnz, d_total_sums, d_total_nnz, *, n_cols, compute_tie_corr, compute_nnz=True, compute_totals=False, col_start=0, col_stop=-1, sub_batch_cols=64)"
)]
pub fn ovr_sparse_csr_host(
    py: Python<'_>,
    h_data: &Bound<'_, PyAny>,
    h_indices: &Bound<'_, PyAny>,
    h_indptr: &Bound<'_, PyAny>,
    h_row_starts: &Bound<'_, PyAny>,
    h_row_stops: &Bound<'_, PyAny>,
    h_group_codes: &Bound<'_, PyAny>,
    h_group_sizes: &Bound<'_, PyAny>,
    d_rank_sums: &Bound<'_, PyAny>,
    d_tie_corr: &Bound<'_, PyAny>,
    d_group_sums: &Bound<'_, PyAny>,
    d_group_nnz: &Bound<'_, PyAny>,
    d_total_sums: &Bound<'_, PyAny>,
    d_total_nnz: &Bound<'_, PyAny>,
    n_cols: usize,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    if h_row_starts.len()? != h_group_codes.len()? || h_row_stops.len()? != h_group_codes.len()? {
        return Err(PyValueError::new_err("row span arrays must match n_rows"));
    }
    ovr_host(
        py,
        h_data,
        h_indices,
        h_indptr,
        h_group_codes,
        h_group_sizes,
        d_rank_sums,
        d_tie_corr,
        d_group_sums,
        d_group_nnz,
        d_total_sums,
        d_total_nnz,
        n_cols,
        compute_tie_corr,
        compute_nnz,
        compute_totals,
        col_start,
        col_stop,
        sub_batch_cols,
        false,
    )
}

fn row_ids<'py>(
    cp: &Bound<'py, PyModule>,
    rows: &Bound<'py, PyAny>,
    mapped: bool,
    n: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let ids = array(cp, rows, "int32", "C")?;
    let ids = if mapped {
        let mask = ids.call_method1("__ge__", (0,))?;
        let selected = cp.call_method1("flatnonzero", (&mask,))?;
        let values = ids.get_item(&selected)?;
        let order = cp.call_method1("argsort", (&values,))?;
        let sorted_positions = values.get_item(&order)?;
        let expected = cp.call_method1("arange", (n,))?;
        if !cp
            .call_method1("array_equal", (&sorted_positions, &expected))?
            .call_method0("item")?
            .extract::<bool>()?
        {
            return Err(PyValueError::new_err(
                "row maps must contain each selected position exactly once",
            ));
        }
        selected.get_item(order)?
    } else {
        ids
    };
    if ids.len()? != n {
        return Err(PyValueError::new_err(
            "row map does not contain the requested number of rows",
        ));
    }
    Ok(ids)
}
fn ovo_device(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    ref_rows: &Bound<'_, PyAny>,
    grp_rows: &Bound<'_, PyAny>,
    grp_offsets: &Bound<'_, PyAny>,
    ranks: &Bound<'_, PyAny>,
    tie: &Bound<'_, PyAny>,
    n_ref: usize,
    n_all_grp: usize,
    compute: bool,
    requested: isize,
    csc: bool,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let dims = shape(ranks)?;
    if dims.len() != 2 {
        return Err(PyValueError::new_err("rank_sums must be two-dimensional"));
    }
    let rows = if csc {
        ref_rows.len()?
    } else {
        indptr
            .len()?
            .checked_sub(1)
            .ok_or_else(|| PyValueError::new_err("empty indptr"))?
    };
    validate_device_sparse(&cp, data, indices, indptr, rows, dims[1], csc, ranks)?;
    let refs = row_ids(&cp, ref_rows, csc, n_ref)?;
    let grps = row_ids(&cp, grp_rows, csc, n_all_grp)?;
    let offsets = wilcoxon::offsets(&cp, grp_offsets, n_all_grp)?;
    if compute {
        let tie = read(tie, &cp, "tie_corr", Some(Dtype::F64), Layout::C)?;
        for object in [data, indices, indptr] {
            tie.require_disjoint(&read(object, &cp, "sparse input", None, Layout::C)?)?;
        }
    }
    let source = if csc {
        None
    } else {
        let kwargs = PyDict::new(py);
        kwargs.set_item("shape", (rows, dims[1]))?;
        kwargs.set_item("copy", false)?;
        Some(
            py.import("cupyx.scipy.sparse")?
                .getattr("csr_matrix")?
                .call(((data, indices, indptr),), Some(&kwargs))?,
        )
    };
    sparse_ovo::ovo(
        &cp,
        rows,
        &refs,
        &grps,
        &offsets,
        ranks,
        tie,
        compute,
        requested,
        None,
        |a, b| {
            if let Some(source) = &source {
                csc_block(&cp, &window(source, a, b)?, false)
            } else {
                Ok(CscBlock {
                    data: data.clone(),
                    indices: indices.clone(),
                    indptr: indptr.get_item(slice(py, a, b + 1))?,
                })
            }
        },
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,ref_rows,grp_rows,grp_offsets,rank_sums,tie_corr,*,n_ref,n_all_grp,compute_tie_corr,sub_batch_cols=64))]
pub fn ovo_streaming_csr_device(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    ref_rows: &Bound<'_, PyAny>,
    grp_rows: &Bound<'_, PyAny>,
    grp_offsets: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    n_ref: usize,
    n_all_grp: usize,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
) -> PyResult<()> {
    ovo_device(
        py,
        data,
        indices,
        indptr,
        ref_rows,
        grp_rows,
        grp_offsets,
        rank_sums,
        tie_corr,
        n_ref,
        n_all_grp,
        compute_tie_corr,
        sub_batch_cols,
        false,
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,ref_rows,grp_rows,grp_offsets,rank_sums,tie_corr,*,n_ref,n_all_grp,compute_tie_corr,sub_batch_cols=64))]
pub fn ovo_streaming_csc_device(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    ref_rows: &Bound<'_, PyAny>,
    grp_rows: &Bound<'_, PyAny>,
    grp_offsets: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    n_ref: usize,
    n_all_grp: usize,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
) -> PyResult<()> {
    ovo_device(
        py,
        data,
        indices,
        indptr,
        ref_rows,
        grp_rows,
        grp_offsets,
        rank_sums,
        tie_corr,
        n_ref,
        n_all_grp,
        compute_tie_corr,
        sub_batch_cols,
        true,
    )
}

fn ovo_host<'py>(
    cp: &Bound<'py, PyModule>,
    source: &Bound<'py, PyAny>,
    refs: &Bound<'py, PyAny>,
    grps: &Bound<'py, PyAny>,
    offsets: &Bound<'py, PyAny>,
    ranks: &Bound<'py, PyAny>,
    tie: &Bound<'py, PyAny>,
    sums: &Bound<'py, PyAny>,
    nnz: &Bound<'py, PyAny>,
    compute: bool,
    compute_nnz: bool,
    requested: isize,
    start: isize,
    stop: isize,
) -> PyResult<()> {
    let dims = shape(source)?;
    let (start, stop) = range(start, stop, dims[1])?;
    if shape(ranks)?.get(1) != Some(&(stop - start)) {
        return Err(PyValueError::new_err(
            "rank output must match column window",
        ));
    }
    let offsets = wilcoxon::offsets(cp, offsets, grps.len()?)?;
    let st = Stats {
        sums,
        nnz: compute_nnz.then_some(nnz),
        total: None,
        total_nnz: None,
    };
    sparse_ovo::ovo(
        cp,
        dims[0],
        refs,
        grps,
        &offsets,
        ranks,
        tie,
        compute,
        requested,
        Some(st),
        |a, b| csc_block(cp, &window(source, start + a, start + b)?, true),
    )
}
#[pyfunction]
#[pyo3(signature=(h_data,h_indices,h_indptr,h_ref_row_map,h_grp_row_map,h_grp_offsets,h_stats_codes,d_rank_sums,d_tie_corr,d_group_sums,d_group_nnz,*,n_ref,n_all_grp,compute_tie_corr,compute_nnz=true,sub_batch_cols=64,analytic_zeros=false))]
pub fn ovo_streaming_csc_host(
    py: Python<'_>,
    h_data: &Bound<'_, PyAny>,
    h_indices: &Bound<'_, PyAny>,
    h_indptr: &Bound<'_, PyAny>,
    h_ref_row_map: &Bound<'_, PyAny>,
    h_grp_row_map: &Bound<'_, PyAny>,
    h_grp_offsets: &Bound<'_, PyAny>,
    h_stats_codes: &Bound<'_, PyAny>,
    d_rank_sums: &Bound<'_, PyAny>,
    d_tie_corr: &Bound<'_, PyAny>,
    d_group_sums: &Bound<'_, PyAny>,
    d_group_nnz: &Bound<'_, PyAny>,
    n_ref: usize,
    n_all_grp: usize,
    compute_tie_corr: bool,
    compute_nnz: bool,
    sub_batch_cols: isize,
    analytic_zeros: bool,
) -> PyResult<()> {
    let _ = analytic_zeros;
    let cp = py.import("cupy")?;
    let rows = h_ref_row_map.len()?;
    if h_stats_codes.len()? != rows || h_grp_row_map.len()? != rows {
        return Err(PyValueError::new_err("row map lengths must match"));
    }
    let cols = h_indptr
        .len()?
        .checked_sub(1)
        .ok_or_else(|| PyValueError::new_err("empty indptr"))?;
    let source = host_sparse(h_data, h_indices, h_indptr, rows, cols, true)?;
    let refs = row_ids(&cp, h_ref_row_map, true, n_ref)?;
    let grps = row_ids(&cp, h_grp_row_map, true, n_all_grp)?;
    ovo_host(
        &cp,
        &source,
        &refs,
        &grps,
        h_grp_offsets,
        d_rank_sums,
        d_tie_corr,
        d_group_sums,
        d_group_nnz,
        compute_tie_corr,
        compute_nnz,
        sub_batch_cols,
        0,
        -1,
    )
}
#[pyfunction]
#[pyo3(signature=(h_data,h_indices,h_row_starts,h_row_stops,h_ref_row_ids,h_grp_row_ids,h_grp_offsets,d_rank_sums,d_tie_corr,d_group_sums,d_group_nnz,*,n_cols,compute_tie_corr,compute_nnz=true,col_start=0,col_stop=-1,sub_batch_cols=64,analytic_zeros=false))]
#[pyo3(
    text_signature = "(h_data, h_indices, h_row_starts, h_row_stops, h_ref_row_ids, h_grp_row_ids, h_grp_offsets, d_rank_sums, d_tie_corr, d_group_sums, d_group_nnz, *, n_cols, compute_tie_corr, compute_nnz=True, col_start=0, col_stop=-1, sub_batch_cols=64, analytic_zeros=False)"
)]
pub fn ovo_streaming_csr_host(
    py: Python<'_>,
    h_data: &Bound<'_, PyAny>,
    h_indices: &Bound<'_, PyAny>,
    h_row_starts: &Bound<'_, PyAny>,
    h_row_stops: &Bound<'_, PyAny>,
    h_ref_row_ids: &Bound<'_, PyAny>,
    h_grp_row_ids: &Bound<'_, PyAny>,
    h_grp_offsets: &Bound<'_, PyAny>,
    d_rank_sums: &Bound<'_, PyAny>,
    d_tie_corr: &Bound<'_, PyAny>,
    d_group_sums: &Bound<'_, PyAny>,
    d_group_nnz: &Bound<'_, PyAny>,
    n_cols: usize,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
    analytic_zeros: bool,
) -> PyResult<()> {
    let _ = analytic_zeros;
    let cp = py.import("cupy")?;
    let np = py.import("numpy")?;
    host_vector(h_row_starts, "h_row_starts", true)?;
    host_vector(h_row_stops, "h_row_stops", true)?;
    let rows = h_row_starts.len()?;
    if h_row_stops.len()? != rows {
        return Err(PyValueError::new_err(
            "row spans must have matching lengths",
        ));
    }
    let first = if rows == 0 {
        0
    } else {
        h_row_starts.get_item(0)?.extract::<usize>()?
    };
    let last = if rows == 0 {
        0
    } else {
        h_row_stops.get_item(rows - 1)?.extract::<usize>()?
    };
    if last < first || last > h_data.len()? {
        return Err(PyValueError::new_err("invalid sparse row spans"));
    }
    let ends = h_row_stops.get_item(slice(py, rows.saturating_sub(1), rows))?;
    let p = if rows == 0 {
        np.call_method1("array", (vec![0i64],))?
    } else {
        np.call_method1("concatenate", ((h_row_starts, &ends),))?
    };
    let p = p.call_method1("__sub__", (first,))?;
    let data = h_data.get_item(slice(py, first, last))?;
    let indices = h_indices.get_item(slice(py, first, last))?;
    let source = host_sparse(&data, &indices, &p, rows, n_cols, false)?;
    let refs = row_ids(&cp, h_ref_row_ids, false, h_ref_row_ids.len()?)?;
    let grps = row_ids(&cp, h_grp_row_ids, false, h_grp_row_ids.len()?)?;
    ovo_host(
        &cp,
        &source,
        &refs,
        &grps,
        h_grp_offsets,
        d_rank_sums,
        d_tie_corr,
        d_group_sums,
        d_group_nnz,
        compute_tie_corr,
        compute_nnz,
        sub_batch_cols,
        col_start,
        col_stop,
    )
}

use pyo3::buffer::PyBuffer;
enum HostIndex {
    I32(PyBuffer<i32>),
    I64(PyBuffer<i64>),
}
impl HostIndex {
    fn read(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name: String = obj.getattr("dtype")?.getattr("name")?.extract()?;
        match name.as_str() {
            "int32" => Ok(Self::I32(PyBuffer::get(obj)?)),
            "int64" => Ok(Self::I64(PyBuffer::get(obj)?)),
            _ => Err(PyTypeError::new_err(
                "host sparse indices must have dtype int32 or int64",
            )),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::I32(b) => b.item_count(),
            Self::I64(b) => b.item_count(),
        }
    }
    fn get(&self, py: Python<'_>, i: usize) -> PyResult<i64> {
        match self {
            Self::I32(b) => b
                .as_slice(py)
                .and_then(|s| s.get(i))
                .map(|v| v.get() as i64),
            Self::I64(b) => b.as_slice(py).and_then(|s| s.get(i)).map(|v| v.get()),
        }
        .ok_or_else(|| PyValueError::new_err("invalid host index buffer or offset"))
    }
    fn set(&self, py: Python<'_>, i: usize, value: i64) -> PyResult<()> {
        match self {
            Self::I32(b) => {
                let v = i32::try_from(value)
                    .map_err(|_| PyValueError::new_err("boundary exceeds int32 range"))?;
                b.as_mut_slice(py)
                    .and_then(|s| s.get(i))
                    .ok_or_else(|| {
                        PyValueError::new_err("boundaries must be writable and contiguous")
                    })?
                    .set(v);
            }
            Self::I64(b) => {
                b.as_mut_slice(py)
                    .and_then(|s| s.get(i))
                    .ok_or_else(|| {
                        PyValueError::new_err("boundaries must be writable and contiguous")
                    })?
                    .set(value);
            }
        }
        Ok(())
    }
}
#[pyfunction]
#[pyo3(signature=(h_indices,h_indptr,h_col_cuts,h_boundaries,*,n_cols))]
pub fn csr_row_boundaries_host(
    py: Python<'_>,
    h_indices: &Bound<'_, PyAny>,
    h_indptr: &Bound<'_, PyAny>,
    h_col_cuts: &Bound<'_, PyAny>,
    h_boundaries: &Bound<'_, PyAny>,
    n_cols: usize,
) -> PyResult<()> {
    host_vector(h_indices, "h_indices", true)?;
    host_vector(h_indptr, "h_indptr", true)?;
    host_vector(h_col_cuts, "h_col_cuts", true)?;
    let indices = HostIndex::read(h_indices)?;
    let indptr = HostIndex::read(h_indptr)?;
    let cuts = HostIndex::read(h_col_cuts)?;
    let out = HostIndex::read(h_boundaries)?;
    let rows = indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| PyValueError::new_err("indptr must not be empty"))?;
    if shape(h_boundaries)? != [cuts.len(), rows]
        || h_boundaries
            .getattr("dtype")?
            .ne(h_indptr.getattr("dtype")?)?
    {
        return Err(PyValueError::new_err(
            "boundaries must have shape (n_cuts,n_rows) and indptr dtype",
        ));
    }
    let np = py.import("numpy")?;
    for input in [h_indices, h_indptr, h_col_cuts] {
        if np
            .call_method1("may_share_memory", (input, h_boundaries))?
            .extract::<bool>()?
        {
            return Err(PyValueError::new_err("boundaries must not overlap inputs"));
        }
    }
    let mut cut_values = Vec::with_capacity(cuts.len());
    for i in 0..cuts.len() {
        let v = cuts.get(py, i)?;
        if v < 0 || v as usize > n_cols || cut_values.last().is_some_and(|last| *last > v) {
            return Err(PyValueError::new_err(
                "cuts must be sorted within [0,n_cols]",
            ));
        }
        cut_values.push(v);
    }
    // Validate every row before writing any output.
    let mut previous = indptr.get(py, 0)?;
    if previous < 0 || previous as usize > indices.len() {
        return Err(PyValueError::new_err("invalid indptr"));
    }
    for i in 1..indptr.len() {
        let v = indptr.get(py, i)?;
        if v < previous || v as usize > indices.len() {
            return Err(PyValueError::new_err(
                "indptr must be monotonic and within indices",
            ));
        }
        previous = v;
    }
    for row in 0..rows {
        let mut start = indptr.get(py, row)? as usize;
        let end = indptr.get(py, row + 1)? as usize;
        for (cut, &value) in cut_values.iter().enumerate() {
            let mut hi = end;
            while start < hi {
                let mid = start + (hi - start) / 2;
                if indices.get(py, mid)? < value {
                    start = mid + 1;
                } else {
                    hi = mid;
                }
            }
            out.set(py, cut * rows + row, start as i64)?;
        }
    }
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(indices,indptr,local_indptr,*,col_start,col_stop,stream=0))]
pub fn csr_column_range_indptr_device(
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    local_indptr: &Bound<'_, PyAny>,
    col_start: u64,
    col_stop: u64,
    stream: usize,
) -> PyResult<i64> {
    if col_stop < col_start || col_stop > i64::MAX as u64 {
        return Err(PyValueError::new_err("invalid column range"));
    }
    let cp = py.import("cupy")?;
    let ix = vector(indices, &cp, "indices", None)?;
    let p = vector(indptr, &cp, "indptr", Some(ix.dtype))?;
    let o = vector(local_indptr, &cp, "local_indptr", Some(p.dtype))?;
    if p.len == 0 || p.len != o.len {
        return Err(PyValueError::new_err(
            "local_indptr must match nonempty indptr",
        ));
    }
    o.require_disjoint(&ix)?;
    o.require_disjoint(&p)?;
    let _scope = StreamScope::new(&cp, stream)?;
    launch(
        &cp,
        "rank_csr_range",
        (p.len - 1).max(1),
        stream,
        &[&ix, &p, &o],
        &mut [
            Arg::P(ix.pointer),
            Arg::P(p.pointer),
            Arg::P(0),
            Arg::P(o.pointer),
            Arg::P(0),
            Arg::P(0),
            Arg::N(p.len - 1),
            Arg::N(ix.len),
            Arg::N(0),
            Arg::N(col_start),
            Arg::N(col_stop),
            Arg::U(integer(&ix)?),
            Arg::U(0),
            Arg::U(0),
        ],
    )?;
    let kw = PyDict::new(py);
    kw.set_item("out", local_indptr)?;
    kw.set_item("dtype", local_indptr.getattr("dtype")?)?;
    cp.getattr("cumsum")?.call((local_indptr,), Some(&kw))?;
    let n = local_indptr
        .get_item(-1)?
        .call_method0("item")?
        .extract::<i64>()?;
    if n < 0 {
        return Err(PyValueError::new_err("local nnz overflow"));
    }
    Ok(n)
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,local_indptr,local_data,local_indices,*,col_start,col_stop,stream=0))]
pub fn csr_column_range_gather_device(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    local_indptr: &Bound<'_, PyAny>,
    local_data: &Bound<'_, PyAny>,
    local_indices: &Bound<'_, PyAny>,
    col_start: u64,
    col_stop: u64,
    stream: usize,
) -> PyResult<()> {
    if col_stop < col_start || col_stop > i64::MAX as u64 {
        return Err(PyValueError::new_err("invalid column range"));
    }
    let cp = py.import("cupy")?;
    let d = floating(data, &cp, "data", Layout::C)?;
    d.require_vector("data")?;
    let ix = vector(indices, &cp, "indices", None)?;
    let p = vector(indptr, &cp, "indptr", Some(ix.dtype))?;
    let lp = vector(local_indptr, &cp, "local_indptr", Some(p.dtype))?;
    let od = vector(local_data, &cp, "local_data", Some(d.dtype))?;
    let oi = vector(local_indices, &cp, "local_indices", Some(ix.dtype))?;
    if p.len == 0 || p.len != lp.len || d.len != ix.len || od.len != oi.len {
        return Err(PyValueError::new_err(
            "inconsistent sparse range buffer lengths",
        ));
    }
    for o in [&od, &oi] {
        for a in [&d, &ix, &p, &lp] {
            o.require_disjoint(a)?;
        }
    }
    od.require_disjoint(&oi)?;
    launch(
        &cp,
        "rank_csr_range",
        product(&[p.len - 1, 32])?,
        stream,
        &[&d, &ix, &p, &lp, &od, &oi],
        &mut [
            Arg::P(ix.pointer),
            Arg::P(p.pointer),
            Arg::P(d.pointer),
            Arg::P(lp.pointer),
            Arg::P(od.pointer),
            Arg::P(oi.pointer),
            Arg::N(p.len - 1),
            Arg::N(d.len),
            Arg::N(od.len),
            Arg::N(col_start),
            Arg::N(col_stop),
            Arg::U(integer(&ix)?),
            Arg::U(wide(d.dtype)?),
            Arg::U(1),
        ],
    )
}
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rapids_singlecell._cuda._wilcoxon_sparse_cuda")?;
    m.setattr("__backend__", "rust")?;
    macro_rules! functions {($($f:ident),+)=>{$(m.add_function(wrap_pyfunction!($f,&m)?)?;)+};}
    functions!(
        csr_row_boundaries_host,
        csr_column_range_indptr_device,
        csr_column_range_gather_device,
        ovr_sparse_csc_device,
        ovr_sparse_csr_device,
        ovr_sparse_csc_host,
        ovr_sparse_csr_host,
        ovo_streaming_csc_device,
        ovo_streaming_csr_device,
        ovo_streaming_csc_host,
        ovo_streaming_csr_host
    );
    m.add_function(wrap_pyfunction!(
        crate::rank_stream::_set_host_worker_limit,
        &m
    )?)?;
    parent.add_submodule(&m)
}
