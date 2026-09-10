//! Bounded host-to-device streaming for aggregation and histograms.
//!
//! Bounded CuPy streams overlap staging and native reductions. Each slot retains
//! its host and device allocations until the prior batch has completed, and
//! all work completes before returning to Python, preserving the host API.
#![allow(non_snake_case, clippy::too_many_arguments)]
mod dense;
use crate::{
    array::{Array, Dtype, Layout},
    rank_support::*,
    staging::BatchStreams,
    wilcoxon_binned,
};
use dense::dense_host;
use pyo3::{
    exceptions::{PyMemoryError, PyTypeError, PyValueError},
    prelude::*,
};
#[pyfunction]
pub fn _set_host_worker_limit(limit: i32) -> i32 {
    crate::host_parallel::set_worker_limit(limit)
}

fn host_array<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    integer: bool,
) -> PyResult<(Vec<usize>, String)> {
    let np = obj.py().import("numpy")?;
    if !obj.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a NumPy host array"
        )));
    }
    let dims = shape(obj)?;
    let dtype: String = obj.getattr("dtype")?.getattr("name")?.extract()?;
    if !(if integer {
        matches!(dtype.as_str(), "int32" | "int64")
    } else {
        matches!(dtype.as_str(), "float32" | "float64")
    }) {
        return Err(PyTypeError::new_err(format!(
            "unsupported {name} dtype: {dtype}"
        )));
    }
    if !obj
        .getattr("dtype")?
        .getattr("isnative")?
        .extract::<bool>()?
    {
        return Err(PyTypeError::new_err(format!(
            "{name} must use native byte order"
        )));
    }
    let flags = obj.getattr("flags")?;
    if !flags.getattr("c_contiguous")?.extract::<bool>()?
        && !flags.getattr("f_contiguous")?.extract::<bool>()?
    {
        return Err(PyValueError::new_err(format!("{name} must be contiguous")));
    }
    Ok((dims, dtype))
}
/// Reserve at most one fifth of currently available device memory across all
/// reusable slots. Private pinned snapshots have the same bounded capacities.
fn staging_budget(cp: &Bound<'_, PyModule>) -> PyResult<u64> {
    let free: (u64, u64) = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("memGetInfo")?
        .extract()?;
    Ok(free.0 / 5)
}

fn dtype_bytes(name: &str) -> u64 {
    if name.ends_with("64") { 8 } else { 4 }
}

fn histogram_window<'py>(
    h: &Histogram<'_, 'py>,
    start: usize,
    stop: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let plane = product(&[h.groups, h.bins + 1])?;
    let first = product(&[(start - h.start) as u64, plane])?;
    let last = product(&[(stop - h.start) as u64, plane])?;
    h.out
        .call_method0("ravel")?
        .get_item(slice(h.out.py(), first as usize, last as usize))
}

fn outputs_present(
    s: Option<&Bound<'_, PyAny>>,
    c: Option<&Bound<'_, PyAny>>,
    q: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if s.is_none() && c.is_none() && q.is_none() {
        return Err(PyValueError::new_err(
            "at least one output plane is required",
        ));
    }
    Ok(())
}

/// Reuse the sparse data in each staging slot; never materialize dense rows.
fn sparse_stats(
    cp: &Bound<'_, PyModule>,
    d: &Bound<'_, PyAny>,
    ix: &Bound<'_, PyAny>,
    p: &Bound<'_, PyAny>,
    codes: &Bound<'_, PyAny>,
    mask: Option<&Bound<'_, PyAny>>,
    sums: Option<&Bound<'_, PyAny>>,
    counts: Option<&Bound<'_, PyAny>>,
    squares: Option<&Bound<'_, PyAny>>,
    rows: u64,
    out_cols: u64,
    col_offset: u64,
    csc: bool,
    stream: usize,
) -> PyResult<()> {
    let d = floating(d, cp, "data", Layout::C)?;
    let ix = vector(ix, cp, "indices", None)?;
    let p = vector(p, cp, "indptr", None)?;
    let c = vector(codes, cp, "cats", Some(Dtype::I32))?;
    if ix.len != d.len || p.len == 0 {
        return Err(PyValueError::new_err("inconsistent sparse block lengths"));
    }
    capacity(&c, rows, "cats")?;
    let major = p.len - 1;
    if (!csc && major != rows)
        || (csc && col_offset.checked_add(major).is_none_or(|e| e > out_cols))
    {
        return Err(PyValueError::new_err("invalid sparse aggregation window"));
    }
    let mask = mask
        .map(|m| vector(m, cp, "mask", Some(Dtype::Bool)))
        .transpose()?;
    if let Some(m) = &mask {
        capacity(m, rows, "mask")?;
    }
    let outs = [sums, counts, squares]
        .into_iter()
        .map(|a| {
            a.map(|a| read(a, cp, "output", Some(Dtype::F64), Layout::C))
                .transpose()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let groups = outs
        .iter()
        .flatten()
        .next()
        .map(|o| o.shape.first().copied().unwrap_or(0) as u64)
        .unwrap_or(0);
    for (i, o) in outs.iter().enumerate() {
        if let Some(o) = o {
            capacity(o, product(&[groups, out_cols])?, "output")?;
            for a in [&d, &ix, &p, &c] {
                o.require_disjoint(a)?;
            }
            if let Some(m) = &mask {
                o.require_disjoint(m)?;
            }
            for q in outs.iter().take(i).flatten() {
                o.require_disjoint(q)?;
            }
        }
    }
    let ptr = |i: usize| outs[i].as_ref().map_or(0, |o| o.pointer);
    let mut arrays = vec![&d, &ix, &p, &c];
    arrays.extend(mask.iter());
    arrays.extend(outs.iter().flatten());
    launch(
        cp,
        "rank_stream_aggr",
        product(&[major, 256])?,
        stream,
        &arrays,
        &mut [
            Arg::P(p.pointer),
            Arg::P(ix.pointer),
            Arg::P(d.pointer),
            Arg::P(c.pointer),
            Arg::P(mask.as_ref().map_or(0, |m| m.pointer)),
            Arg::P(ptr(0)),
            Arg::P(ptr(1)),
            Arg::P(ptr(2)),
            Arg::N(major),
            Arg::N(rows),
            Arg::N(out_cols),
            Arg::N(groups),
            Arg::N(col_offset),
            Arg::N(d.len),
            Arg::U(integer(&p)?),
            Arg::U(integer(&ix)?),
            Arg::U(wide(d.dtype)?),
            Arg::U(csc as u32),
        ],
    )
}

struct Histogram<'a, 'py> {
    out: &'a Bound<'py, PyAny>,
    groups: u64,
    bins: u64,
    low: f64,
    inverse: f64,
    start: usize,
    stop: usize,
}

/// Validate shared operands once; every streaming batch borrows these same
/// caller-owned allocations until its completion guard has drained all work.
struct ValidatedStream {
    codes: Array,
    mask: Option<Array>,
    outputs: Vec<Option<Array>>,
    histogram: Option<Array>,
    groups: u64,
}

fn validate_stream(
    cp: &Bound<'_, PyModule>,
    codes: &Bound<'_, PyAny>,
    mask: Option<&Bound<'_, PyAny>>,
    outputs: [Option<&Bound<'_, PyAny>>; 3],
    rows: usize,
    cols: usize,
    hist: Option<&Histogram<'_, '_>>,
) -> PyResult<ValidatedStream> {
    product(&[rows as u64, cols as u64])?;
    let codes = vector(codes, cp, "group codes", Some(Dtype::I32))?;
    capacity(&codes, rows as u64, "group codes")?;
    let mask = mask
        .map(|m| vector(m, cp, "mask", Some(Dtype::Bool)))
        .transpose()?;
    if let Some(m) = &mask {
        capacity(m, rows as u64, "mask")?;
    }
    let outputs = outputs
        .into_iter()
        .map(|o| {
            o.map(|o| read(o, cp, "output", Some(Dtype::F64), Layout::C))
                .transpose()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut groups = None;
    for (i, output) in outputs.iter().enumerate() {
        if let Some(output) = output {
            let (n, c) = matrix(output, "output")?;
            if c != cols as u64 || groups.is_some_and(|g| g != n) {
                return Err(PyValueError::new_err(
                    "all output planes must have shape (n_groups, n_genes)",
                ));
            }
            groups = Some(n);
            output.require_disjoint(&codes)?;
            if let Some(m) = &mask {
                output.require_disjoint(m)?;
            }
            for other in outputs.iter().take(i).flatten() {
                output.require_disjoint(other)?;
            }
        }
    }
    let histogram = hist
        .map(|h| {
            if h.bins == 0
                || h.bins > i32::MAX as u64
                || !h.low.is_finite()
                || !h.inverse.is_finite()
            {
                return Err(PyValueError::new_err("invalid histogram bin configuration"));
            }
            let a = read(h.out, cp, "hist", Some(Dtype::U32), Layout::C)?;
            capacity(
                &a,
                product(&[(h.stop - h.start) as u64, h.groups, h.bins + 1])?,
                "hist",
            )?;
            a.require_disjoint(&codes)?;
            for out in outputs.iter().flatten() {
                a.require_disjoint(out)?;
            }
            Ok(a)
        })
        .transpose()?;
    let mut arrays = vec![&codes];
    arrays.extend(mask.iter());
    arrays.extend(outputs.iter().flatten());
    arrays.extend(histogram.iter());
    crate::array::current_device(cp, &arrays)?;
    Ok(ValidatedStream {
        codes,
        mask,
        outputs,
        histogram,
        groups: groups.unwrap_or(0),
    })
}

fn sparse_host<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    sums: Option<&Bound<'py, PyAny>>,
    counts: Option<&Bound<'py, PyAny>>,
    squares: Option<&Bound<'py, PyAny>>,
    mask: Option<&Bound<'py, PyAny>>,
    rows: usize,
    cols: usize,
    requested: isize,
    csc: bool,
    hist: Option<Histogram<'_, 'py>>,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let (ds, dtype) = host_array(data, "data", false)?;
    let (xs, itype) = host_array(indices, "indices", true)?;
    let (ps, ptype) = host_array(indptr, "indptr", true)?;
    let segments = if csc { cols } else { rows };
    if ds.len() != 1 || xs != ds || ps != [segments + 1] {
        return Err(PyValueError::new_err(
            "host sparse lengths do not match matrix dimensions",
        ));
    }
    validate_stream(
        &cp,
        codes,
        mask,
        [sums, counts, squares],
        rows,
        cols,
        hist.as_ref(),
    )?;
    let pointers = crate::host_buffer::sparse_offsets(py, indptr, ds[0])?;
    let first = if csc {
        hist.as_ref().map_or(0, |h| h.start)
    } else {
        0
    };
    let last = if csc {
        hist.as_ref().map_or(cols, |h| h.stop)
    } else {
        rows
    };
    if first == last {
        return Ok(());
    }
    let budget = staging_budget(&cp)?;
    // Histograms use one index-width flag for both compressed arrays. Promote
    // only that path; standalone aggregation supports independent widths.
    let histogram_i64 = hist.is_some() && (itype == "int64" || ptype == "int64");
    let pointer_bytes = if histogram_i64 {
        8
    } else {
        dtype_bytes(&ptype)
    };
    let value_bytes = dtype_bytes(&dtype)
        + if histogram_i64 {
            8
        } else {
            dtype_bytes(&itype)
        };
    let max_segment = pointers[first..=last]
        .windows(2)
        .map(|p| (p[1] - p[0]) as u64)
        .max()
        .unwrap_or(0);
    let minimum = product(&[max_segment, value_bytes])?
        .checked_add(2 * pointer_bytes)
        .ok_or_else(|| PyValueError::new_err("sparse staging size overflow"))?;
    if minimum > budget {
        return Err(PyMemoryError::new_err(
            "one sparse segment exceeds the streaming device-memory budget",
        ));
    }
    let slots = if minimum <= budget / 2 && last - first > 1 {
        2
    } else {
        1
    };
    let per_slot = budget / slots as u64;
    // Reserve the maximum pointer allocation independently of nonzero counts.
    // Reused data/index/pointer capacities can peak in different batches.
    let max_segments = (per_slot.saturating_sub(product(&[max_segment, value_bytes])?)
        / pointer_bytes)
        .saturating_sub(1)
        .min(1 << 20) as usize;
    let sub = (if requested <= 0 {
        4096
    } else {
        requested as usize
    })
    .clamp(1, max_segments.max(1))
    .min(last - first);
    let pointer_budget = product(&[(sub + 1) as u64, pointer_bytes])?;
    let cap = ((per_slot - pointer_budget) / value_bytes).min(usize::MAX as u64) as usize;
    if csc && hist.is_none() {
        for a in [sums, counts, squares].into_iter().flatten() {
            a.call_method1("fill", (0,))?;
        }
    }
    let mut streams = BatchStreams::new(&cp, slots)?;
    let mut start = first;
    let mut batch_i = 0;
    while start < last {
        let base = pointers[start] as usize;
        let mut stop = start + 1;
        while stop < last && stop - start < sub && (pointers[stop + 1] as usize - base) <= cap {
            stop += 1;
        }
        let end = pointers[stop] as usize;
        if end - base > cap {
            return Err(PyMemoryError::new_err(
                "one sparse segment exceeds the streaming device-memory budget",
            ));
        }
        let slot = batch_i % slots;
        let _scope = streams.enter(slot)?;
        let stream = stream(&cp)?;
        let hd = data.get_item(slice(py, base, end))?;
        let hi = indices.get_item(slice(py, base, end))?;
        let hp = indptr
            .get_item(slice(py, start, stop + 1))?
            .call_method1("__sub__", (base,))?;
        let hi = if histogram_i64 && itype != "int64" {
            hi.call_method1("astype", ("int64",))?
        } else {
            hi
        };
        let hp = if histogram_i64 && ptype != "int64" {
            hp.call_method1("astype", ("int64",))?
        } else {
            hp
        };
        let dd = streams.upload_vector(slot, 0, &hd)?;
        let di = streams.upload_vector(slot, 1, &hi)?;
        let dp = streams.upload_vector(slot, 2, &hp)?;
        let local_codes = if csc {
            codes.clone()
        } else {
            codes.get_item(slice(py, start, stop))?
        };
        let local_mask = mask
            .map(|m| {
                if csc {
                    Ok(m.clone())
                } else {
                    m.get_item(slice(py, start, stop))
                }
            })
            .transpose()?;
        streams.retain(slot, local_codes.clone());
        if let Some(a) = &local_mask {
            streams.retain(slot, a.clone());
        }
        let block_rows = if csc { rows } else { stop - start };
        if let Some(h) = &hist {
            let out = if csc {
                histogram_window(h, start, stop)?
            } else {
                h.out.clone()
            };
            streams.retain(slot, out.clone());
            wilcoxon_binned::histogram_sparse(
                &cp,
                &dd,
                &di,
                &dp,
                &local_codes,
                &out,
                block_rows as u64,
                if csc {
                    (stop - start) as u64
                } else {
                    (h.stop - h.start) as u64
                },
                h.groups,
                h.bins,
                h.low,
                h.inverse,
                if csc { 0 } else { h.start as u64 },
                csc,
                stream,
            )?;
        }
        if sums.is_some() || counts.is_some() || squares.is_some() {
            sparse_stats(
                &cp,
                &dd,
                &di,
                &dp,
                &local_codes,
                local_mask.as_ref(),
                sums,
                counts,
                squares,
                block_rows as u64,
                cols as u64,
                if csc { start as u64 } else { 0 },
                csc,
                stream,
            )?;
        }
        start = stop;
        batch_i += 1;
    }
    streams.finish()
}

#[pyfunction]
#[pyo3(signature=(data,indices,indptr,cats,*,out_sum=None,out_count=None,out_sqsum=None,mask=None,n_cells,n_genes,sub_batch_rows=4096))]
fn aggr_csr_host<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    cats: &Bound<'py, PyAny>,
    out_sum: Option<&Bound<'py, PyAny>>,
    out_count: Option<&Bound<'py, PyAny>>,
    out_sqsum: Option<&Bound<'py, PyAny>>,
    mask: Option<&Bound<'py, PyAny>>,
    n_cells: usize,
    n_genes: usize,
    sub_batch_rows: isize,
) -> PyResult<()> {
    outputs_present(out_sum, out_count, out_sqsum)?;
    sparse_host(
        py,
        data,
        indices,
        indptr,
        cats,
        out_sum,
        out_count,
        out_sqsum,
        mask,
        n_cells,
        n_genes,
        sub_batch_rows,
        false,
        None,
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,cats,*,out_sum=None,out_count=None,out_sqsum=None,mask=None,n_cells,n_genes,sub_batch_cols=4096))]
fn aggr_csc_host<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    cats: &Bound<'py, PyAny>,
    out_sum: Option<&Bound<'py, PyAny>>,
    out_count: Option<&Bound<'py, PyAny>>,
    out_sqsum: Option<&Bound<'py, PyAny>>,
    mask: Option<&Bound<'py, PyAny>>,
    n_cells: usize,
    n_genes: usize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    outputs_present(out_sum, out_count, out_sqsum)?;
    sparse_host(
        py,
        data,
        indices,
        indptr,
        cats,
        out_sum,
        out_count,
        out_sqsum,
        mask,
        n_cells,
        n_genes,
        sub_batch_cols,
        true,
        None,
    )
}
#[pyfunction]
#[pyo3(signature=(X,cats,*,out_sum=None,out_count=None,out_sqsum=None,mask=None,sub_batch=4096))]
fn aggr_dense_host<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    cats: &Bound<'py, PyAny>,
    out_sum: Option<&Bound<'py, PyAny>>,
    out_count: Option<&Bound<'py, PyAny>>,
    out_sqsum: Option<&Bound<'py, PyAny>>,
    mask: Option<&Bound<'py, PyAny>>,
    sub_batch: isize,
) -> PyResult<()> {
    outputs_present(out_sum, out_count, out_sqsum)?;
    dense_host(
        py, X, cats, out_sum, out_count, out_sqsum, mask, sub_batch, None,
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,gcodes,hist,*,group_sums=None,group_nnz=None,n_cells,n_genes,n_groups,n_bins,bin_low,inv_bin_width,col_start=0,col_stop=-1,sub_batch_rows=4096))]
#[pyo3(
    text_signature = "(data, indices, indptr, gcodes, hist, *, group_sums=None, group_nnz=None, n_cells, n_genes, n_groups, n_bins, bin_low, inv_bin_width, col_start=0, col_stop=-1, sub_batch_rows=4096)"
)]
fn hist_csr_host<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    gcodes: &Bound<'py, PyAny>,
    hist: &Bound<'py, PyAny>,
    group_sums: Option<&Bound<'py, PyAny>>,
    group_nnz: Option<&Bound<'py, PyAny>>,
    n_cells: usize,
    n_genes: usize,
    n_groups: u64,
    n_bins: u64,
    bin_low: f64,
    inv_bin_width: f64,
    col_start: isize,
    col_stop: isize,
    sub_batch_rows: isize,
) -> PyResult<()> {
    let (start, stop) = range(col_start, col_stop, n_genes)?;
    sparse_host(
        py,
        data,
        indices,
        indptr,
        gcodes,
        group_sums,
        group_nnz,
        None,
        None,
        n_cells,
        n_genes,
        sub_batch_rows,
        false,
        Some(Histogram {
            out: hist,
            groups: n_groups,
            bins: n_bins,
            low: bin_low,
            inverse: inv_bin_width,
            start,
            stop,
        }),
    )
}
#[pyfunction]
#[pyo3(signature=(data,indices,indptr,gcodes,hist,*,group_sums=None,group_nnz=None,n_cells,n_genes,n_groups,n_bins,bin_low,inv_bin_width,col_start,col_stop,sub_batch_cols=4096))]
fn hist_csc_host<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    gcodes: &Bound<'py, PyAny>,
    hist: &Bound<'py, PyAny>,
    group_sums: Option<&Bound<'py, PyAny>>,
    group_nnz: Option<&Bound<'py, PyAny>>,
    n_cells: usize,
    n_genes: usize,
    n_groups: u64,
    n_bins: u64,
    bin_low: f64,
    inv_bin_width: f64,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    let (start, stop) = range(col_start, col_stop, n_genes)?;
    sparse_host(
        py,
        data,
        indices,
        indptr,
        gcodes,
        group_sums,
        group_nnz,
        None,
        None,
        n_cells,
        n_genes,
        sub_batch_cols,
        true,
        Some(Histogram {
            out: hist,
            groups: n_groups,
            bins: n_bins,
            low: bin_low,
            inverse: inv_bin_width,
            start,
            stop,
        }),
    )
}
#[pyfunction]
#[pyo3(signature=(X,gcodes,hist,*,group_sums=None,group_nnz=None,n_groups,n_bins,bin_low,inv_bin_width,col_start,col_stop,sub_batch_cols=4096))]
fn hist_dense_host<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    gcodes: &Bound<'py, PyAny>,
    hist: &Bound<'py, PyAny>,
    group_sums: Option<&Bound<'py, PyAny>>,
    group_nnz: Option<&Bound<'py, PyAny>>,
    n_groups: u64,
    n_bins: u64,
    bin_low: f64,
    inv_bin_width: f64,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    let dims = shape(X)?;
    if dims.len() != 2 {
        return Err(PyValueError::new_err("X must be two-dimensional"));
    }
    let (start, stop) = range(col_start, col_stop, dims[1])?;
    dense_host(
        py,
        X,
        gcodes,
        group_sums,
        group_nnz,
        None,
        None,
        sub_batch_cols,
        Some(Histogram {
            out: hist,
            groups: n_groups,
            bins: n_bins,
            low: bin_low,
            inverse: inv_bin_width,
            start,
            stop,
        }),
    )
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rapids_singlecell._cuda._rank_stream_cuda")?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(_set_host_worker_limit, &m)?)?;
    m.add_function(wrap_pyfunction!(aggr_csr_host, &m)?)?;
    m.add_function(wrap_pyfunction!(aggr_csc_host, &m)?)?;
    m.add_function(wrap_pyfunction!(aggr_dense_host, &m)?)?;
    m.add_function(wrap_pyfunction!(hist_csr_host, &m)?)?;
    m.add_function(wrap_pyfunction!(hist_csc_host, &m)?)?;
    m.add_function(wrap_pyfunction!(hist_dense_host, &m)?)?;
    parent.add_submodule(&m)
}
