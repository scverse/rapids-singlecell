//! Exact Wilcoxon ranks using bounded native GPU batches and stable radix sorting.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Array, Dtype, Layout, current_device},
    rank_support::*,
    runtime::StreamScope,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

pub struct Stats<'a, 'py> {
    pub sums: &'a Bound<'py, PyAny>,
    pub nnz: Option<&'a Bound<'py, PyAny>>,
    pub total: Option<&'a Bound<'py, PyAny>>,
    pub total_nnz: Option<&'a Bound<'py, PyAny>>,
}
fn stats_outputs(
    cp: &Bound<'_, PyModule>,
    stats: &Stats<'_, '_>,
    groups: u64,
    cols: u64,
) -> PyResult<Vec<Array>> {
    let mut arrays = Vec::new();
    for (obj, n) in [
        (Some(stats.sums), groups * cols),
        (stats.nnz, groups * cols),
        (stats.total, cols),
        (stats.total_nnz, cols),
    ] {
        if let Some(obj) = obj {
            let a = read(obj, cp, "statistics output", Some(Dtype::F64), Layout::C)?;
            capacity(&a, n, "statistics output")?;
            current_device(cp, &[&a])?;
            arrays.push(a);
        }
    }
    Ok(arrays)
}
fn rank_outputs(
    cp: &Bound<'_, PyModule>,
    ranks: &Bound<'_, PyAny>,
    tie: &Bound<'_, PyAny>,
    compute: bool,
) -> PyResult<(Array, Option<Array>, usize, usize)> {
    let r = read(ranks, cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (g, c) = matrix(&r, "rank_sums")?;
    let t = if compute {
        let t = read(tie, cp, "tie_corr", Some(Dtype::F64), Layout::C)?;
        capacity(&t, c, "tie_corr")?;
        t.require_disjoint(&r)?;
        Some(t)
    } else {
        None
    };
    current_device(cp, &[&r])?;
    if let Some(t) = &t {
        current_device(cp, &[t])?;
    }
    Ok((r, t, g as usize, c as usize))
}
fn sorted<'py>(
    cp: &Bound<'py, PyModule>,
    x: &Bound<'py, PyAny>,
    indices: bool,
) -> PyResult<Bound<'py, PyAny>> {
    crate::rank_sort::sort(cp, x, indices)
}
/// Accumulate ranks into full-width outputs while staging only a column batch.
pub fn ovr<'py>(
    cp: &Bound<'py, PyModule>,
    rows: usize,
    codes: &Bound<'py, PyAny>,
    ranks: &Bound<'py, PyAny>,
    tie: &Bound<'py, PyAny>,
    compute: bool,
    requested: isize,
    statistics: Option<Stats<'_, 'py>>,
    mut tile: impl FnMut(usize, usize) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<()> {
    let s = stream(cp)?;
    let (r, t, groups, cols) = rank_outputs(cp, ranks, tie, compute)?;
    let c = vector(codes, cp, "group_codes", Some(Dtype::I32))?;
    if c.len != rows as u64 {
        return Err(PyValueError::new_err(
            "group_codes length must equal input rows",
        ));
    }
    r.require_disjoint(&c)?;
    if let Some(t) = &t {
        t.require_disjoint(&c)?;
    }
    current_device(cp, &[&r, &c])?;
    let stats_arrays = if let Some(st) = &statistics {
        stats_outputs(cp, st, groups as u64, cols as u64)?
    } else {
        Vec::new()
    };
    let mut outputs = vec![&r];
    outputs.extend(t.iter());
    outputs.extend(stats_arrays.iter());
    crate::harmony::disjoint(&outputs, &[&c])?;
    for output in &outputs {
        zero(cp, output, s)?;
    }
    let width = batch(cp, rows, requested, cols)?;
    for start in (0..cols).step_by(width) {
        let stop = (start + width).min(cols);
        let original = tile(start, stop)?;
        let x = array(cp, &original, "float32", "F")?;
        let xa = floating(&x, cp, "block", Layout::F)?;
        if xa.shape != [rows, stop - start] {
            return Err(PyValueError::new_err("rank tile has incorrect shape"));
        }
        let order = sorted(cp, &x, true)?;
        let oa = read(&order, cp, "sorted indices", Some(Dtype::I64), Layout::F)?;
        launch(
            cp,
            "rank_sorted",
            xa.len,
            s,
            &[&xa, &oa, &c, &r],
            &mut [
                Arg::P(xa.pointer),
                Arg::P(oa.pointer),
                Arg::P(c.pointer),
                Arg::P(r.pointer),
                Arg::P(t.as_ref().map_or(0, |a| a.pointer)),
                Arg::N(rows as u64),
                Arg::N((stop - start) as u64),
                Arg::N(groups as u64),
                Arg::N(cols as u64),
                Arg::N(start as u64),
            ],
        )?;
        if let Some(st) = &statistics {
            let data = array(cp, &original, "float64", "F")?;
            stats(
                cp,
                &data,
                codes,
                None,
                Some(st.sums),
                None,
                st.nnz,
                st.total,
                st.total_nnz,
                cols as u64,
                start as u64,
                s,
            )?;
        }
    }
    if let Some(t) = &t {
        launch(
            cp,
            "rank_tie_finish",
            cols as u64,
            s,
            &[t],
            &mut [Arg::P(t.pointer), Arg::N(rows as u64), Arg::N(cols as u64)],
        )?;
    }
    // Public streaming entry points historically complete before returning.
    sync(cp)
}

pub fn offsets(
    cp: &Bound<'_, PyModule>,
    object: &Bound<'_, PyAny>,
    rows: usize,
) -> PyResult<Vec<usize>> {
    let numpy = cp.py().import("numpy")?;
    let host = if object.is_instance(&cp.getattr("ndarray")?)? {
        cp.call_method1("asnumpy", (object,))?
    } else {
        object.clone()
    };
    if !host.is_instance(&numpy.getattr("ndarray")?)?
        || host.getattr("ndim")?.extract::<usize>()? != 1
    {
        return Err(PyTypeError::new_err(
            "grp_offsets must be a one-dimensional integer array",
        ));
    }
    let dtype: String = host.getattr("dtype")?.getattr("name")?.extract()?;
    if dtype != "int32" && dtype != "int64" {
        return Err(PyTypeError::new_err(
            "grp_offsets must have dtype int32 or int64",
        ));
    }
    let v = host.call_method0("tolist")?.extract::<Vec<usize>>()?;
    if v.is_empty() || v[0] != 0 || *v.last().unwrap() != rows || v.windows(2).any(|w| w[1] < w[0])
    {
        return Err(PyValueError::new_err(
            "grp_offsets must partition all group rows in sorted order",
        ));
    }
    Ok(v)
}

/// Rank each group against a reference sorted once per column batch.
pub fn ovo<'py>(
    cp: &Bound<'py, PyModule>,
    nref: usize,
    offsets: &[usize],
    ranks: &Bound<'py, PyAny>,
    tie: &Bound<'py, PyAny>,
    compute: bool,
    requested: isize,
    statistics: Option<Stats<'_, 'py>>,
    mut tiles: impl FnMut(usize, usize) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<()> {
    let s = stream(cp)?;
    let (r, t, groups, cols) = rank_outputs(cp, ranks, tie, compute)?;
    if offsets.len() != groups + 1 {
        return Err(PyValueError::new_err(
            "grp_offsets must have n_groups + 1 entries",
        ));
    }
    if let Some(t) = &t
        && t.shape != [groups, cols]
    {
        return Err(PyValueError::new_err(
            "tie_corr must have shape (n_groups, n_cols)",
        ));
    }
    let stats_arrays = if let Some(st) = &statistics {
        stats_outputs(cp, st, (groups + 1) as u64, cols as u64)?
    } else {
        Vec::new()
    };
    let mut outputs = vec![&r];
    outputs.extend(t.iter());
    outputs.extend(stats_arrays.iter());
    crate::harmony::disjoint(&outputs, &[])?;
    for output in &outputs {
        zero(cp, output, s)?;
    }
    let rows = nref + offsets.last().copied().unwrap_or(0);
    let width = batch(cp, rows, requested, cols)?.min(65_535);
    for start in (0..cols).step_by(width) {
        let stop = (start + width).min(cols);
        let (ref_original, grp_original) = tiles(start, stop)?;
        let reference = array(cp, &ref_original, "float32", "F")?;
        let reference = sorted(cp, &reference, false)?;
        let ra = floating(&reference, cp, "ref_data", Layout::F)?;
        if ra.shape != [nref, stop - start] {
            return Err(PyValueError::new_err("reference tile has incorrect shape"));
        }
        for g in 0..groups {
            let group = grp_original
                .get_item((slice(cp.py(), offsets[g], offsets[g + 1]), all(cp.py())))?;
            let group = array(cp, &group, "float32", "F")?;
            let group = sorted(cp, &group, false)?;
            let ga = floating(&group, cp, "grp_data", Layout::F)?;
            let ng = offsets[g + 1] - offsets[g];
            launch(
                cp,
                "rank_ovo",
                ((nref + ng)
                    .div_ceil(256)
                    .max(1)
                    .min(65_535 / (stop - start))
                    * (stop - start)
                    * 256) as u64,
                s,
                &[&ra, &ga, &r],
                &mut [
                    Arg::P(ra.pointer),
                    Arg::P(ga.pointer),
                    Arg::P(r.pointer + (g * cols * 8) as u64),
                    Arg::P(t.as_ref().map_or(0, |t| t.pointer + (g * cols * 8) as u64)),
                    Arg::N(nref as u64),
                    Arg::N(ng as u64),
                    Arg::N((stop - start) as u64),
                    Arg::N(cols as u64),
                    Arg::N(start as u64),
                ],
            )?;
        }
        if let Some(st) = &statistics {
            let kw = PyDict::new(cp.py());
            kw.set_item("dtype", "int32")?;
            let mut labels = Vec::with_capacity(rows);
            for g in 0..groups {
                labels.extend(std::iter::repeat_n(g as i32, offsets[g + 1] - offsets[g]));
            }
            labels.extend(std::iter::repeat_n(groups as i32, nref));
            let codes = cp.getattr("asarray")?.call((labels,), Some(&kw))?;
            let group_data = array(cp, &grp_original, "float64", "F")?;
            let reference_data = array(cp, &ref_original, "float64", "F")?;
            let combined = cp.call_method1("concatenate", ((group_data, reference_data),))?;
            let data = array(cp, &combined, "float64", "F")?;
            stats(
                cp,
                &data,
                &codes,
                None,
                Some(st.sums),
                None,
                st.nnz,
                None,
                None,
                cols as u64,
                start as u64,
                s,
            )?;
        }
    }
    if let Some(t) = &t {
        for g in 0..groups {
            launch(
                cp,
                "rank_tie_finish",
                cols as u64,
                s,
                &[t],
                &mut [
                    Arg::P(t.pointer + (g * cols * 8) as u64),
                    Arg::N((nref + offsets[g + 1] - offsets[g]) as u64),
                    Arg::N(cols as u64),
                ],
            )?;
        }
    }
    sync(cp)
}
pub fn host_dense(object: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    let np = object.py().import("numpy")?;
    if !object.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err("X must be a NumPy array"));
    }
    let dims = shape(object)?;
    let dtype: String = object.getattr("dtype")?.getattr("name")?.extract()?;
    let f = object.getattr("flags")?;
    if dims.len() != 2
        || !(f.getattr("c_contiguous")?.extract::<bool>()?
            || f.getattr("f_contiguous")?.extract::<bool>()?)
    {
        return Err(PyValueError::new_err(
            "X must be a contiguous two-dimensional array",
        ));
    }
    if dtype != "float32" && dtype != "float64" {
        return Err(PyTypeError::new_err("X must have dtype float32 or float64"));
    }
    Ok((dims[0], dims[1]))
}
#[pyfunction]
#[pyo3(signature=(block,group_codes,rank_sums,tie_corr,*,compute_tie_corr,sub_batch_cols=64,stream=0))]
pub fn ovr_rank_dense_streaming(
    py: Python<'_>,
    block: &Bound<'_, PyAny>,
    group_codes: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let x = read(block, &cp, "block", Some(Dtype::F32), Layout::F)?;
    let (rows, cols) = matrix(&x, "block")?;
    let r = read(rank_sums, &cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (_, out_cols) = matrix(&r, "rank_sums")?;
    if cols != out_cols {
        return Err(PyValueError::new_err(
            "rank output columns must match input",
        ));
    }
    r.require_disjoint(&x)?;
    current_device(&cp, &[&x, &r])?;
    if compute_tie_corr {
        let t = read(tie_corr, &cp, "tie_corr", Some(Dtype::F64), Layout::C)?;
        t.require_disjoint(&x)?;
    }

    let _scope = StreamScope::new(&cp, stream)?;
    ovr(
        &cp,
        rows as usize,
        group_codes,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        None,
        |a, b| window(block, a, b),
    )
}
#[pyfunction]
#[pyo3(signature=(X,group_codes,rank_sums,tie_corr,group_sums,group_nnz,total_sums,total_nnz,*,compute_tie_corr,compute_nnz,compute_totals,col_start=0,col_stop=-1,sub_batch_cols=64), text_signature="(X,group_codes,rank_sums,tie_corr,group_sums,group_nnz,total_sums,total_nnz,*,compute_tie_corr,compute_nnz,compute_totals,col_start=0,col_stop=-1,sub_batch_cols=64)")]
pub fn ovr_rank_dense_host_streaming(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    group_codes: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    group_sums: &Bound<'_, PyAny>,
    group_nnz: &Bound<'_, PyAny>,
    total_sums: &Bound<'_, PyAny>,
    total_nnz: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    compute_nnz: bool,
    compute_totals: bool,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let (rows, cols) = host_dense(X)?;
    let (start, stop) = range(col_start, col_stop, cols)?;
    if shape(rank_sums)?.get(1) != Some(&(stop - start)) {
        return Err(PyValueError::new_err(
            "rank output must match column window",
        ));
    }
    let st = Stats {
        sums: group_sums,
        nnz: compute_nnz.then_some(group_nnz),
        total: compute_totals.then_some(total_sums),
        total_nnz: (compute_totals && compute_nnz).then_some(total_nnz),
    };
    ovr(
        &cp,
        rows,
        group_codes,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        Some(st),
        |a, b| window(X, start + a, start + b),
    )
}
#[pyfunction]
#[pyo3(signature=(ref_data,grp_data,grp_offsets,rank_sums,tie_corr,*,compute_tie_corr,sub_batch_cols=64,stream=0))]
pub fn ovo_rank_dense_tiered_unsorted_ref(
    py: Python<'_>,
    ref_data: &Bound<'_, PyAny>,
    grp_data: &Bound<'_, PyAny>,
    grp_offsets: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    sub_batch_cols: isize,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let ra = read(ref_data, &cp, "ref_data", Some(Dtype::F32), Layout::F)?;
    let ga = read(grp_data, &cp, "grp_data", Some(Dtype::F32), Layout::F)?;
    let (rows, cols) = matrix(&ra, "ref_data")?;
    let (ng, gc) = matrix(&ga, "grp_data")?;
    let r = read(rank_sums, &cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (_, rc) = matrix(&r, "rank_sums")?;
    if cols != gc || cols != rc {
        return Err(PyValueError::new_err(
            "all dense rank buffers must have matching columns",
        ));
    }
    r.require_disjoint(&ra)?;
    r.require_disjoint(&ga)?;
    current_device(&cp, &[&ra, &ga, &r])?;
    if compute_tie_corr {
        let t = read(tie_corr, &cp, "tie_corr", Some(Dtype::F64), Layout::C)?;
        t.require_disjoint(&ra)?;
        t.require_disjoint(&ga)?;
    }

    let _scope = StreamScope::new(&cp, stream)?;
    let off = offsets(&cp, grp_offsets, ng as usize)?;
    ovo(
        &cp,
        rows as usize,
        &off,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        None,
        |a, b| Ok((window(ref_data, a, b)?, window(grp_data, a, b)?)),
    )
}
/// Validate host gather indices before rank/statistics outputs are cleared.
fn host_row_ids(ids: &Bound<'_, PyAny>, rows: usize) -> PyResult<usize> {
    let np = ids.py().import("numpy")?;
    if !ids.is_instance(&np.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err("row ids must be NumPy arrays"));
    }
    let dims = shape(ids)?;
    let dtype: String = ids.getattr("dtype")?.getattr("name")?.extract()?;
    if dims.len() != 1 || (dtype != "int32" && dtype != "int64") {
        return Err(PyTypeError::new_err(
            "row ids must be one-dimensional int32 or int64 arrays",
        ));
    }
    if dims[0] != 0 {
        let low = ids.call_method0("min")?.extract::<i64>()?;
        let high = ids.call_method0("max")?.extract::<i64>()?;
        if low < 0 || high as u64 >= rows as u64 {
            return Err(PyValueError::new_err(
                "row ids must be within the input row range",
            ));
        }
    }
    Ok(dims[0])
}
#[pyfunction]
#[pyo3(signature=(X,ref_row_ids,grp_row_ids,grp_offsets,rank_sums,tie_corr,group_sums,group_nnz,*,compute_tie_corr,compute_nnz,col_start=0,col_stop=-1,sub_batch_cols=64), text_signature="(X,ref_row_ids,grp_row_ids,grp_offsets,rank_sums,tie_corr,group_sums,group_nnz,*,compute_tie_corr,compute_nnz,col_start=0,col_stop=-1,sub_batch_cols=64)")]
pub fn ovo_rank_dense_host_streaming(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    ref_row_ids: &Bound<'_, PyAny>,
    grp_row_ids: &Bound<'_, PyAny>,
    grp_offsets: &Bound<'_, PyAny>,
    rank_sums: &Bound<'_, PyAny>,
    tie_corr: &Bound<'_, PyAny>,
    group_sums: &Bound<'_, PyAny>,
    group_nnz: &Bound<'_, PyAny>,
    compute_tie_corr: bool,
    compute_nnz: bool,
    col_start: isize,
    col_stop: isize,
    sub_batch_cols: isize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let (rows, cols) = host_dense(X)?;
    let (start, stop) = range(col_start, col_stop, cols)?;
    let ref_rows = host_row_ids(ref_row_ids, rows)?;
    let group_rows = host_row_ids(grp_row_ids, rows)?;
    let off = offsets(&cp, grp_offsets, group_rows)?;
    if shape(rank_sums)?.get(1) != Some(&(stop - start)) {
        return Err(PyValueError::new_err(
            "rank output must match column window",
        ));
    }
    let st = Stats {
        sums: group_sums,
        nnz: compute_nnz.then_some(group_nnz),
        total: None,
        total_nnz: None,
    };
    ovo(
        &cp,
        ref_rows,
        &off,
        rank_sums,
        tie_corr,
        compute_tie_corr,
        sub_batch_cols,
        Some(st),
        |a, b| {
            let columns = slice(py, start + a, start + b);
            Ok((
                X.get_item((ref_row_ids, columns.clone()))?,
                X.get_item((grp_row_ids, columns))?,
            ))
        },
    )
}
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "rapids_singlecell._cuda._wilcoxon_cuda")?;
    m.setattr("__backend__", "rust")?;
    m.add_function(wrap_pyfunction!(
        crate::rank_stream::_set_host_worker_limit,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(ovr_rank_dense_streaming, &m)?)?;
    m.add_function(wrap_pyfunction!(ovr_rank_dense_host_streaming, &m)?)?;
    m.add_function(wrap_pyfunction!(ovo_rank_dense_tiered_unsorted_ref, &m)?)?;
    m.add_function(wrap_pyfunction!(ovo_rank_dense_host_streaming, &m)?)?;
    parent.add_submodule(&m)
}
