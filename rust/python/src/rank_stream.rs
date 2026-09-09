//! Bounded host-to-device streaming for aggregation and histograms.
//!
//! Two CuPy streams overlap staging and native reductions. Each slot retains
//! its host and device allocations until the prior batch has completed, and
//! all work completes before returning to Python, preserving the host API.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Dtype, Layout},
    rank_support::*,
    runtime, wilcoxon_binned,
};
use pyo3::{
    exceptions::{PyMemoryError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::cell::Cell;
thread_local! {static HOST_WORKERS:Cell<i32>=const{Cell::new(0)};}
#[pyfunction]
pub fn _set_host_worker_limit(limit: i32) -> i32 {
    HOST_WORKERS.with(|v| v.replace(limit.max(0)))
}

struct Slot<'py> {
    stream: Bound<'py, PyAny>,
    pending: Vec<Py<PyAny>>,
}
impl Slot<'_> {
    fn prepare(&mut self) -> PyResult<usize> {
        self.stream.call_method0("synchronize")?;
        self.pending.clear();
        self.stream.getattr("ptr")?.extract()
    }
    fn keep(&mut self, a: &Bound<'_, PyAny>) {
        self.pending.push(a.clone().unbind());
    }
}
struct Streams<'py> {
    slots: Vec<Slot<'py>>,
}
impl<'py> Streams<'py> {
    fn new(cp: &Bound<'py, PyModule>) -> PyResult<Self> {
        sync(cp)?;
        let kw = PyDict::new(cp.py());
        kw.set_item("non_blocking", true)?;
        let factory = cp.getattr("cuda")?.getattr("Stream")?;
        let mut slots = Vec::new();
        for _ in 0..2 {
            slots.push(Slot {
                stream: factory.call((), Some(&kw))?,
                pending: Vec::new(),
            });
        }
        Ok(Self { slots })
    }
    fn finish(&mut self) -> PyResult<()> {
        for slot in &mut self.slots {
            slot.prepare()?;
        }
        Ok(())
    }
}
impl Drop for Streams<'_> {
    fn drop(&mut self) {
        for slot in &self.slots {
            let _ = slot.stream.call_method0("synchronize");
        }
    }
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
    let flags = obj.getattr("flags")?;
    if !flags.getattr("c_contiguous")?.extract::<bool>()?
        && !flags.getattr("f_contiguous")?.extract::<bool>()?
    {
        return Err(PyValueError::new_err(format!("{name} must be contiguous")));
    }
    Ok((dims, dtype))
}
fn nnz_budget(cp: &Bound<'_, PyModule>, bytes: u64) -> PyResult<usize> {
    let free: (u64, u64) = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("memGetInfo")?
        .extract()?;
    Ok((free.0 / 5 / bytes.max(1)).clamp(1, 2_000_000_000) as usize)
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
fn validate_stream(
    cp: &Bound<'_, PyModule>,
    codes: &Bound<'_, PyAny>,
    mask: Option<&Bound<'_, PyAny>>,
    outputs: [Option<&Bound<'_, PyAny>>; 3],
    rows: usize,
    cols: usize,
    hist: Option<&Histogram<'_, '_>>,
) -> PyResult<()> {
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
            if h.bins == 0 || !h.low.is_finite() || !h.inverse.is_finite() {
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
    Ok(())
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
    let index_dtype = if itype == "int64" || ptype == "int64" {
        "int64"
    } else {
        "int32"
    };
    // Reading the host indptr is safe and bounds are checked before every slice.
    let pointers: Vec<i64> = indptr.call_method0("tolist")?.extract()?;
    if pointers.iter().any(|&p| p < 0 || p as usize > ds[0])
        || pointers.windows(2).any(|w| w[0] > w[1])
    {
        return Err(PyValueError::new_err("invalid compressed sparse indptr"));
    }
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
    let sub = (if requested <= 0 {
        4096
    } else {
        requested as usize
    })
    .clamp(1, 1 << 20);
    let cap = nnz_budget(
        &cp,
        if dtype == "float64" { 8 } else { 4 } + if index_dtype == "int64" { 8 } else { 4 },
    )?;
    if csc && hist.is_none() {
        for a in [sums, counts, squares].into_iter().flatten() {
            a.call_method1("fill", (0,))?;
        }
    }
    let mut streams = Streams::new(&cp)?;
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
        let slot = &mut streams.slots[batch_i % 2];
        let stream = slot.prepare()?;
        let _scope = runtime::StreamScope::new(&cp, stream)?;
        let hd = data.get_item(slice(py, base, end))?;
        let hi = indices.get_item(slice(py, base, end))?;
        let hp = indptr
            .get_item(slice(py, start, stop + 1))?
            .call_method1("__sub__", (base,))?;
        let dd = array(&cp, &hd, &dtype, "C")?;
        let di = array(&cp, &hi, index_dtype, "C")?;
        let dp = array(&cp, &hp, index_dtype, "C")?;
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
        let block_rows = if csc { rows } else { stop - start };
        if let Some(h) = &hist {
            let out = if csc {
                h.out.get_item(slice(py, start - h.start, stop - h.start))?
            } else {
                h.out.clone()
            };
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
        for a in [&hd, &hi, &hp, &dd, &di, &dp, &local_codes] {
            slot.keep(a);
        }
        if let Some(a) = &local_mask {
            slot.keep(a);
        }
        start = stop;
        batch_i += 1;
    }
    streams.finish()
}

fn dense_host<'py>(
    py: Python<'py>,
    X: &Bound<'py, PyAny>,
    codes: &Bound<'py, PyAny>,
    sums: Option<&Bound<'py, PyAny>>,
    counts: Option<&Bound<'py, PyAny>>,
    squares: Option<&Bound<'py, PyAny>>,
    mask: Option<&Bound<'py, PyAny>>,
    requested: isize,
    hist: Option<Histogram<'_, 'py>>,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let (dims, dtype) = host_array(X, "X", false)?;
    if dims.len() != 2 {
        return Err(PyValueError::new_err("X must be two-dimensional"));
    }
    let rows = dims[0];
    let cols = dims[1];
    validate_stream(
        &cp,
        codes,
        mask,
        [sums, counts, squares],
        rows,
        cols,
        hist.as_ref(),
    )?;
    let fortran = X
        .getattr("flags")?
        .getattr("f_contiguous")?
        .extract::<bool>()?;
    let columns = fortran || hist.is_some();
    let first = hist.as_ref().map_or(0, |h| h.start);
    let last = if columns {
        hist.as_ref().map_or(cols, |h| h.stop)
    } else {
        rows
    };
    let sub = batch(
        &cp,
        if columns { rows } else { cols },
        if requested <= 0 { 4096 } else { requested },
        last - first,
    )?;
    if fortran && hist.is_none() {
        for a in [sums, counts, squares].into_iter().flatten() {
            a.call_method1("fill", (0,))?;
        }
    }
    if let Some(h) = &hist {
        for a in [sums, counts].into_iter().flatten() {
            a.get_item((all(py), slice(py, h.start, h.stop)))?
                .call_method1("fill", (0,))?;
        }
    }
    let mut streams = Streams::new(&cp)?;
    let mut start = first;
    let mut batch_i = 0;
    while start < last {
        let stop = (start + sub).min(last);
        let slot = &mut streams.slots[batch_i % 2];
        let stream = slot.prepare()?;
        let _scope = runtime::StreamScope::new(&cp, stream)?;
        let hx = if columns {
            window(X, start, stop)?
        } else {
            X.get_item((slice(py, start, stop), all(py)))?
        };
        let dx = array(&cp, &hx, &dtype, "F")?;
        let local_codes = if columns {
            codes.clone()
        } else {
            codes.get_item(slice(py, start, stop))?
        };
        let local_mask = mask
            .map(|m| {
                if columns {
                    Ok(m.clone())
                } else {
                    m.get_item(slice(py, start, stop))
                }
            })
            .transpose()?;
        if let Some(h) = &hist {
            let out = h.out.get_item(slice(py, start - h.start, stop - h.start))?;
            wilcoxon_binned::histogram_dense(
                &cp,
                &dx,
                &local_codes,
                &out,
                rows as u64,
                (stop - start) as u64,
                h.groups,
                h.bins,
                h.low,
                h.inverse,
                false,
                stream,
            )?;
        }
        if sums.is_some() || counts.is_some() || squares.is_some() {
            stats(
                &cp,
                &dx,
                &local_codes,
                local_mask.as_ref(),
                sums,
                squares,
                counts,
                None,
                None,
                cols as u64,
                if columns { start as u64 } else { 0 },
                stream,
            )?;
        }
        for a in [&hx, &dx, &local_codes] {
            slot.keep(a);
        }
        if let Some(a) = &local_mask {
            slot.keep(a);
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
