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
    types::{PyDict, PySlice},
};

/// Validated sparse host storage with one reusable row-span plan per call.
enum HostSource<'py> {
    Csc(Bound<'py, PyAny>),
    Csr {
        data: Bound<'py, PyAny>,
        indices: Bound<'py, PyAny>,
        spans: crate::host_sparse::CsrSpans,
        shape: [usize; 2],
    },
}

impl<'py> HostSource<'py> {
    fn csr(
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        indices: &Bound<'py, PyAny>,
        starts: &Bound<'py, PyAny>,
        stops: &Bound<'py, PyAny>,
        columns: usize,
    ) -> PyResult<Self> {
        host_vector(data, "data", false)?;
        host_vector(indices, "indices", true)?;
        if data.len()? != indices.len()? {
            return Err(PyValueError::new_err(
                "CSR data and indices lengths must match",
            ));
        }
        let spans = crate::host_sparse::CsrSpans::new(py, indices, starts, stops, columns)?;
        Ok(Self::Csr {
            data: data.clone(),
            indices: indices.clone(),
            spans,
            shape: [starts.len()?, columns],
        })
    }

    fn shape(&self) -> PyResult<[usize; 2]> {
        match self {
            Self::Csc(source) => {
                let (rows, cols) = source.getattr("shape")?.extract()?;
                Ok([rows, cols])
            }
            Self::Csr { shape, .. } => Ok(*shape),
        }
    }

    fn window(&self, first: usize, stop: usize) -> PyResult<CscBlock<'py>> {
        match self {
            Self::Csc(source) => host_csc_window(source, first, stop),
            Self::Csr {
                data,
                indices,
                spans,
                ..
            } => spans
                .window(data.py(), data, indices, first, stop)
                .map(CscBlock::Owned),
        }
    }
}

/// Slice validated host CSC storage without SciPy's redundant value/index copy.
/// The staging layer takes the owned snapshot required before detaching.
fn host_csc_window<'py>(
    source: &Bound<'py, PyAny>,
    first: usize,
    stop: usize,
) -> PyResult<CscBlock<'py>> {
    let py = source.py();
    let indptr = source.getattr("indptr")?;
    let begin = indptr.get_item(first)?.extract::<isize>()?;
    let end = indptr.get_item(stop)?.extract::<isize>()?;
    let entries = PySlice::new(py, begin, end, 1);
    let pointers = PySlice::new(py, first as isize, stop as isize + 1, 1);
    let data = source.getattr("data")?.get_item(&entries)?;
    let indices = source.getattr("indices")?.get_item(&entries)?;
    let indptr = indptr
        .get_item(pointers)?
        .call_method1("__sub__", (begin,))?;
    crate::host_sparse::CscWindow::from_arrays(py, &data, &indices, &indptr).map(CscBlock::Owned)
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
/// Validate host CSC metadata once while borrowing the original NumPy arrays.
fn host_csc<'py>(
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    host_vector(data, "data", false)?;
    host_vector(indices, "indices", true)?;
    host_vector(indptr, "indptr", true)?;
    if data.len()? != indices.len()? || indptr.len()? != cols + 1 {
        return Err(PyValueError::new_err("inconsistent sparse array lengths"));
    }
    let kw = PyDict::new(data.py());
    kw.set_item("shape", (rows, cols))?;
    kw.set_item("copy", false)?;
    let matrix = data
        .py()
        .import("scipy.sparse")?
        .getattr("csc_matrix")?
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
    let mut csr_windows = source
        .as_ref()
        .map(|source| crate::device_sparse::CsrWindows::new(&cp, source, c.len as usize, out[1]))
        .transpose()?;
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
        false,
        |a, b, staging, slot| {
            if let Some(windows) = &mut csr_windows {
                windows.window(&cp, a, b, staging, slot)
            } else {
                Ok(CscBlock::arrays(
                    data.clone(),
                    indices.clone(),
                    indptr.get_item(slice(py, a, b + 1))?,
                ))
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
    source: &HostSource<'_>,
    codes: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    ranks: &Bound<'_, PyAny>,
    tie: &Bound<'_, PyAny>,
    sums: &Bound<'_, PyAny>,
    nnz: &Bound<'_, PyAny>,
    total: &Bound<'_, PyAny>,
    total_nnz: &Bound<'_, PyAny>,
    compute: bool,
    compute_nnz: bool,
    compute_totals: bool,
    start: isize,
    stop: isize,
    batch: isize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    host_vector(codes, "group_codes", true)?;
    let rows = codes.len()?;
    let dimensions = source.shape()?;
    if dimensions[0] != rows {
        return Err(PyValueError::new_err(
            "group codes must match the source rows",
        ));
    }
    let (start, stop) = range(start, stop, dimensions[1])?;
    let out = shape(ranks)?;
    if out != [sizes.len()?, stop - start] {
        return Err(PyValueError::new_err(
            "rank_sums must have shape (n_groups, window_cols)",
        ));
    }
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
        true,
        |a, b, _, _| source.window(start + a, start + b),
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
    let source = HostSource::Csc(host_csc(
        h_data,
        h_indices,
        h_indptr,
        h_group_codes.len()?,
        cols,
    )?);
    ovr_host(
        py,
        &source,
        h_group_codes,
        h_group_sizes,
        d_rank_sums,
        d_tie_corr,
        d_group_sums,
        d_group_nnz,
        d_total_sums,
        d_total_nnz,
        compute_tie_corr,
        compute_nnz,
        compute_totals,
        0,
        -1,
        sub_batch_cols,
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
    host_vector(h_indptr, "h_indptr", true)?;
    if h_indptr.len()? != h_group_codes.len()? + 1 {
        return Err(PyValueError::new_err(
            "CSR indptr must have n_rows+1 entries",
        ));
    }
    crate::host_buffer::sparse_offsets(py, h_indptr, h_data.len()?)?;
    let source = HostSource::csr(py, h_data, h_indices, h_row_starts, h_row_stops, n_cols)?;
    ovr_host(
        py,
        &source,
        h_group_codes,
        h_group_sizes,
        d_rank_sums,
        d_tie_corr,
        d_group_sums,
        d_group_nnz,
        d_total_sums,
        d_total_nnz,
        compute_tie_corr,
        compute_nnz,
        compute_totals,
        col_start,
        col_stop,
        sub_batch_cols,
    )
}

fn row_ids<'py>(
    cp: &Bound<'py, PyModule>,
    rows: &Bound<'py, PyAny>,
    mapped: bool,
    n: usize,
    completion: &mut crate::staging::BatchStreams<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let ids = array(cp, rows, "int32", "C")?;
    completion.retain(0, ids.clone());
    let ids = if mapped {
        let mask = ids.call_method1("__ge__", (0,))?;
        completion.retain(0, mask.clone());
        let selected = cp.call_method1("flatnonzero", (&mask,))?;
        completion.retain(0, selected.clone());
        let values = ids.get_item(&selected)?;
        completion.retain(0, values.clone());
        let order = cp.call_method1("argsort", (&values,))?;
        completion.retain(0, order.clone());
        let sorted_positions = values.get_item(&order)?;
        completion.retain(0, sorted_positions.clone());
        let expected = cp.call_method1("arange", (n,))?;
        completion.retain(0, expected.clone());
        let equal = cp.call_method1("array_equal", (&sorted_positions, &expected))?;
        completion.retain(0, equal.clone());
        if !equal.call_method0("item")?.extract::<bool>()? {
            return Err(PyValueError::new_err(
                "row maps must contain each selected position exactly once",
            ));
        }
        let selected = selected.get_item(order)?;
        completion.retain(0, selected.clone());
        selected
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
    // Selection casts and row-map gathering precede the main sparse pipeline.
    // Retain each temporary as it is created so early callback/allocation
    // failures still drain the caller stream before releasing GPU storage.
    let mut completion = crate::staging::BatchStreams::caller(&cp)?;
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
    let refs = row_ids(&cp, ref_rows, csc, n_ref, &mut completion)?;
    let grps = row_ids(&cp, grp_rows, csc, n_all_grp, &mut completion)?;
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
    let mut csr_windows = source
        .as_ref()
        .map(|source| crate::device_sparse::CsrWindows::new(&cp, source, rows, dims[1]))
        .transpose()?;
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
        None,
        false,
        |_| Ok(None),
        |a, b, staging, slot| {
            if let Some(windows) = &mut csr_windows {
                windows.window(&cp, a, b, staging, slot)
            } else {
                Ok(CscBlock::arrays(
                    data.clone(),
                    indices.clone(),
                    indptr.get_item(slice(py, a, b + 1))?,
                ))
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
    source: &HostSource<'py>,
    refs: &Bound<'py, PyAny>,
    grps: &Bound<'py, PyAny>,
    offsets: &Bound<'py, PyAny>,
    ranks: &Bound<'py, PyAny>,
    tie: &Bound<'py, PyAny>,
    sums: &Bound<'py, PyAny>,
    nnz: &Bound<'py, PyAny>,
    stat_codes: Option<&Bound<'py, PyAny>>,
    compute: bool,
    compute_nnz: bool,
    requested: isize,
    start: isize,
    stop: isize,
) -> PyResult<()> {
    let dims = source.shape()?;
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
        stat_codes,
        true,
        |populations| match source {
            HostSource::Csr {
                data,
                indices,
                spans,
                ..
            } => spans.dense_rank_pack(cp.py(), data, indices, populations, start, stop),
            HostSource::Csc(_) => Ok(None),
        },
        |a, b, _, _| source.window(start + a, start + b),
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
    let source = HostSource::Csc(host_csc(h_data, h_indices, h_indptr, rows, cols)?);
    let numpy = py.import("numpy")?;
    let refs = numpy.call_method1(
        "asarray",
        (
            crate::host_sparse::mapped_rows(py, h_ref_row_map, n_ref)?,
            "int64",
        ),
    )?;
    let grps = numpy.call_method1(
        "asarray",
        (
            crate::host_sparse::mapped_rows(py, h_grp_row_map, n_all_grp)?,
            "int64",
        ),
    )?;
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
        Some(h_stats_codes),
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
    let source = HostSource::csr(py, h_data, h_indices, h_row_starts, h_row_stops, n_cols)?;
    ovo_host(
        &cp,
        &source,
        h_ref_row_ids,
        h_grp_row_ids,
        h_grp_offsets,
        d_rank_sums,
        d_tie_corr,
        d_group_sums,
        d_group_nnz,
        None,
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
enum OwnedIndex {
    I32(Vec<i32>),
    I64(Vec<i64>),
}
impl Default for OwnedIndex {
    fn default() -> Self {
        Self::I32(Vec::new())
    }
}
impl OwnedIndex {
    fn bytes(&self) -> usize {
        match self {
            Self::I32(v) => v.capacity() * 4,
            Self::I64(v) => v.capacity() * 8,
        }
    }
}
#[derive(Default)]
struct BoundaryWorkspace {
    indices: OwnedIndex,
    indptr: OwnedIndex,
    cuts: Vec<i64>,
    output: OwnedIndex,
}
thread_local! {
    // Retain only one bounded workspace per caller, avoiding repeated large
    // allocation/page-fault costs when planning successive windows.
    static BOUNDARY_WORKSPACE: std::cell::RefCell<Option<BoundaryWorkspace>> = const { std::cell::RefCell::new(None) };
}
const MAX_BOUNDARY_CACHE_BYTES: usize = 128 * 1024 * 1024;

impl HostIndex {
    fn read(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name: String = obj.getattr("dtype")?.getattr("name")?.extract()?;
        let buffer = match name.as_str() {
            "int32" => Self::I32(PyBuffer::get(obj)?),
            "int64" => Self::I64(PyBuffer::get(obj)?),
            _ => {
                return Err(PyTypeError::new_err(
                    "host sparse indices must have dtype int32 or int64",
                ));
            }
        };
        let contiguous = match &buffer {
            Self::I32(b) => b.is_c_contiguous(),
            Self::I64(b) => b.is_c_contiguous(),
        };
        if !contiguous {
            return Err(PyValueError::new_err("host indices must be C-contiguous"));
        }
        Ok(buffer)
    }
    fn len(&self) -> usize {
        match self {
            Self::I32(b) => b.item_count(),
            Self::I64(b) => b.item_count(),
        }
    }
    fn snapshot_into(&self, py: Python<'_>, destination: &mut OwnedIndex) -> PyResult<()> {
        match self {
            Self::I32(buffer) => {
                if !matches!(destination, OwnedIndex::I32(_)) {
                    *destination = OwnedIndex::I32(Vec::new());
                }
                let OwnedIndex::I32(values) = destination else {
                    unreachable!()
                };
                values.resize(buffer.item_count(), 0);
                buffer.copy_to_slice(py, values)
            }
            Self::I64(buffer) => {
                if !matches!(destination, OwnedIndex::I64(_)) {
                    *destination = OwnedIndex::I64(Vec::new());
                }
                let OwnedIndex::I64(values) = destination else {
                    unreachable!()
                };
                values.resize(buffer.item_count(), 0);
                buffer.copy_to_slice(py, values)
            }
        }
    }
    fn writable(&self, py: Python<'_>) -> bool {
        match self {
            Self::I32(b) => b.as_mut_slice(py).is_some(),
            Self::I64(b) => b.as_mut_slice(py).is_some(),
        }
    }
}
trait BoundaryIndex: Copy + Default + Send + Sync + Into<i64> {
    fn offset(value: usize) -> Self;
}
impl BoundaryIndex for i32 {
    fn offset(value: usize) -> Self {
        value as i32
    }
}
impl BoundaryIndex for i64 {
    fn offset(value: usize) -> Self {
        value as i64
    }
}

fn find_boundaries<I: BoundaryIndex, P: BoundaryIndex>(
    indices: &[I],
    indptr: &[P],
    cuts: &[i64],
    n_cols: usize,
    plan: crate::host_parallel::Parallelism,
    output: &mut Vec<P>,
) -> PyResult<()> {
    if cuts.iter().any(|&cut| cut < 0 || cut as usize > n_cols)
        || cuts.windows(2).any(|w| w[0] > w[1])
    {
        return Err(PyValueError::new_err(
            "cuts must be sorted within [0,n_cols]",
        ));
    }
    // Complete validation precedes all output writes, including for empty cuts.
    let mut previous = 0;
    for &pointer in indptr {
        let value = pointer.into();
        if value < previous || value as usize > indices.len() {
            return Err(PyValueError::new_err(
                "indptr must be monotonic and within indices",
            ));
        }
        previous = value;
    }
    let rows = indptr.len() - 1;
    let count = rows
        .checked_mul(cuts.len())
        .ok_or_else(|| PyValueError::new_err("boundary shape overflow"))?;
    output
        .try_reserve_exact(count.saturating_sub(output.len()))
        .map_err(|_| {
            pyo3::exceptions::PyMemoryError::new_err("unable to allocate boundary workspace")
        })?;
    output.resize(count, P::default());
    if rows == 0 || cuts.is_empty() {
        return Ok(());
    }

    // Partition every column into the same disjoint row spans. This preserves
    // the legacy advancing lower bound across sorted cuts without raw pointers
    // or a row-major temporary and transpose.
    let chunk = rows.div_ceil(plan.workers());
    let mut partitions: Vec<_> = (0..rows)
        .step_by(chunk)
        .map(|start| (start, Vec::with_capacity(cuts.len())))
        .collect();
    for column in output.chunks_mut(rows) {
        for ((_, spans), part) in partitions.iter_mut().zip(column.chunks_mut(chunk)) {
            spans.push(part);
        }
    }
    plan.for_each_chunk(&mut partitions, |_, parts| {
        for (row_start, spans) in parts {
            for local_row in 0..spans[0].len() {
                let row = *row_start + local_row;
                let base = indptr[row].into() as usize;
                let end = indptr[row + 1].into() as usize;
                let values = &indices[base..end];
                let mut start = 0;
                if cuts.len() >= values.len().div_ceil(16) {
                    // Many cuts over a short sparse row are cheaper as one
                    // ordered merge than repeated binary searches.
                    for (&cut, column) in cuts.iter().zip(spans.iter_mut()) {
                        while start < values.len() && values[start].into() < cut {
                            start += 1;
                        }
                        column[local_row] = P::offset(base + start);
                    }
                } else {
                    for (&cut, column) in cuts.iter().zip(spans.iter_mut()) {
                        start += values[start..].partition_point(|&value| value.into() < cut);
                        // The result lies between validated indptr entries and
                        // therefore fits the original pointer dtype.
                        column[local_row] = P::offset(base + start);
                    }
                }
            }
        }
    });
    Ok(())
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
    if n_cols > i64::MAX as usize {
        return Err(PyValueError::new_err("n_cols exceeds int64 range"));
    }
    if shape(h_boundaries)? != [cuts.len(), rows]
        || h_boundaries
            .getattr("dtype")?
            .ne(h_indptr.getattr("dtype")?)?
    {
        return Err(PyValueError::new_err(
            "boundaries must have shape (n_cuts,n_rows) and indptr dtype",
        ));
    }
    if !out.writable(py) {
        return Err(PyValueError::new_err(
            "boundaries must be writable and contiguous",
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
    // NumPy may be changed by another Python thread while detached. Workers
    // therefore only see immutable Rust-owned snapshots and private output.
    // Keep the output buffer exported until the final attached copy, preventing
    // normal NumPy resize operations from invalidating its storage.
    let mut workspace = BOUNDARY_WORKSPACE
        .with(|cache| cache.borrow_mut().take())
        .unwrap_or_default();
    indices.snapshot_into(py, &mut workspace.indices)?;
    indptr.snapshot_into(py, &mut workspace.indptr)?;
    workspace.cuts.clear();
    match &cuts {
        HostIndex::I32(buffer) => workspace.cuts.extend(
            buffer
                .as_slice(py)
                .unwrap()
                .iter()
                .map(|value| i64::from(value.get())),
        ),
        HostIndex::I64(buffer) => workspace
            .cuts
            .extend(buffer.as_slice(py).unwrap().iter().map(|value| value.get())),
    }
    if matches!(indptr, HostIndex::I32(_)) && !matches!(workspace.output, OwnedIndex::I32(_)) {
        workspace.output = OwnedIndex::I32(Vec::new());
    }
    if matches!(indptr, HostIndex::I64(_)) && !matches!(workspace.output, OwnedIndex::I64(_)) {
        workspace.output = OwnedIndex::I64(Vec::new());
    }
    let result = crate::host_parallel::run(py, rows, |plan| {
        match (&workspace.indices, &workspace.indptr, &mut workspace.output) {
            (OwnedIndex::I32(i), OwnedIndex::I32(p), OwnedIndex::I32(o)) => {
                find_boundaries(i, p, &workspace.cuts, n_cols, plan, o)
            }
            (OwnedIndex::I64(i), OwnedIndex::I32(p), OwnedIndex::I32(o)) => {
                find_boundaries(i, p, &workspace.cuts, n_cols, plan, o)
            }
            (OwnedIndex::I32(i), OwnedIndex::I64(p), OwnedIndex::I64(o)) => {
                find_boundaries(i, p, &workspace.cuts, n_cols, plan, o)
            }
            (OwnedIndex::I64(i), OwnedIndex::I64(p), OwnedIndex::I64(o)) => {
                find_boundaries(i, p, &workspace.cuts, n_cols, plan, o)
            }
            _ => unreachable!("boundary workspace dtype matches indptr"),
        }
    })
    .and_then(|result| result)
    .and_then(|()| match (&out, &workspace.output) {
        (HostIndex::I32(buffer), OwnedIndex::I32(values)) => buffer.copy_from_slice(py, values),
        (HostIndex::I64(buffer), OwnedIndex::I64(values)) => buffer.copy_from_slice(py, values),
        _ => unreachable!("boundary dtype was validated against indptr"),
    });
    if workspace.indices.bytes()
        + workspace.indptr.bytes()
        + workspace.output.bytes()
        + workspace.cuts.capacity() * 8
        <= MAX_BOUNDARY_CACHE_BYTES
    {
        BOUNDARY_WORKSPACE.with(|cache| *cache.borrow_mut() = Some(workspace));
    }
    result
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
