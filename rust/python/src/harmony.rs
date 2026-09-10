//! Native Harmony bindings and shared, validated launch primitives.
#![allow(non_snake_case, clippy::too_many_arguments)]
use crate::{
    array::{Array, Dtype, Layout, current_device},
    runtime,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};
use std::ffi::c_void;

pub(crate) enum Arg {
    Ptr(u64),
    U64(u64),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
}
impl Arg {
    pub(crate) fn pointer(&mut self) -> *mut c_void {
        match self {
            Self::Ptr(x) | Self::U64(x) => (x as *mut u64).cast(),
            Self::U32(x) => (x as *mut u32).cast(),
            Self::I32(x) => (x as *mut i32).cast(),
            Self::F32(x) => (x as *mut f32).cast(),
            Self::F64(x) => (x as *mut f64).cast(),
        }
    }
}
pub(crate) fn scalar(dtype: Dtype, x: f64) -> Arg {
    if dtype == Dtype::F32 {
        Arg::F32(x as f32)
    } else {
        Arg::F64(x)
    }
}
pub(crate) fn launch(
    device: usize,
    name: &str,
    dtype: Dtype,
    work: u64,
    stream: usize,
    args: &mut [Arg],
) -> PyResult<()> {
    launch_grid(device, name, dtype, work.div_ceil(256), 256, stream, args)
}

pub(crate) fn pen_kernel(covariates: u64) -> &'static str {
    match covariates {
        1 => "harmony_pen_norm_cov1",
        2 => "harmony_pen_norm_cov2",
        3 => "harmony_pen_norm_cov3",
        4 => "harmony_pen_norm_cov4",
        _ => "harmony_pen_norm",
    }
}

pub(crate) fn launch_rows(
    device: usize,
    name: &str,
    dtype: Dtype,
    rows: u64,
    cols: u64,
    stream: usize,
    args: &mut [Arg],
) -> PyResult<()> {
    let block = cols.clamp(1, 256).div_ceil(32) * 32;
    launch_grid(device, name, dtype, rows, block, stream, args)
}

pub(crate) fn launch_grid(
    device: usize,
    name: &str,
    dtype: Dtype,
    blocks: u64,
    block: u64,
    stream: usize,
    args: &mut [Arg],
) -> PyResult<()> {
    if blocks == 0 {
        return Ok(());
    }
    let suffix = match dtype {
        Dtype::F32 => "f32",
        Dtype::F64 => "f64",
        Dtype::I32 => "i32",
        _ => return Err(PyTypeError::new_err("unsupported kernel dtype")),
    };
    let name = format!("{name}_{suffix}");
    let mut pointers: Vec<_> = args.iter_mut().map(Arg::pointer).collect();
    // SAFETY: each binding validates typed allocation bounds and dimensions;
    // argument storage lives through CUDA's synchronous argument copy.
    unsafe {
        runtime::launch(
            device,
            &name,
            (blocks.min(65535) as u32, 1, 1),
            (block as u32, 1, 1),
            stream,
            &mut pointers,
        )
    }
}
pub(crate) fn read(
    obj: &Bound<'_, PyAny>,
    cp: &Bound<'_, PyModule>,
    name: &str,
    dtype: Option<Dtype>,
    minimum: u64,
) -> PyResult<Array> {
    let a = Array::read(obj, cp, name, dtype, None, Layout::C)?;
    if a.len < minimum {
        return Err(PyValueError::new_err(format!(
            "{name} requires at least {minimum} elements"
        )));
    }
    Ok(a)
}
pub(crate) fn floating(
    obj: &Bound<'_, PyAny>,
    cp: &Bound<'_, PyModule>,
    name: &str,
    minimum: u64,
) -> PyResult<Array> {
    let a = read(obj, cp, name, None, minimum)?;
    if !matches!(a.dtype, Dtype::F32 | Dtype::F64) {
        return Err(PyTypeError::new_err(format!(
            "{name} must have dtype float32 or float64"
        )));
    }
    Ok(a)
}
pub(crate) fn product(dims: &[u64]) -> PyResult<u64> {
    dims.iter()
        .try_fold(1u64, |a, &b| {
            a.checked_mul(b).filter(|&n| n <= isize::MAX as u64 / 8)
        })
        .ok_or_else(|| PyValueError::new_err("dimensions exceed addressable memory"))
}
pub(crate) fn zero(cp: &Bound<'_, PyModule>, ptr: u64, bytes: u64, stream: usize) -> PyResult<()> {
    cp.getattr("cuda")?
        .getattr("runtime")?
        .call_method1("memsetAsync", (ptr, 0, bytes, stream))?;
    Ok(())
}
pub(crate) fn copy(
    cp: &Bound<'_, PyModule>,
    dst: u64,
    src: u64,
    bytes: u64,
    stream: usize,
) -> PyResult<()> {
    cp.getattr("cuda")?
        .getattr("runtime")?
        .call_method1("memcpyAsync", (dst, src, bytes, 3, stream))?;
    Ok(())
}
pub(crate) fn itemsize(dtype: Dtype) -> u64 {
    if dtype == Dtype::F64 { 8 } else { 4 }
}

#[pyfunction]
#[pyo3(signature=(v,*,cats,n_cells,n_pcs,n_covariates=1,switcher,a,stream=0))]
pub fn scatter_add(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    n_cells: u64,
    n_pcs: u64,
    n_covariates: u64,
    switcher: i32,
    a: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    if n_covariates == 0 {
        return Err(PyValueError::new_err(
            "scatter_add requires at least one covariate",
        ));
    }
    let cp = py.import("cupy")?;
    let v = floating(v, &cp, "v", product(&[n_cells, n_pcs])?)?;
    let cats = read(
        cats,
        &cp,
        "cats",
        Some(Dtype::I32),
        product(&[n_cells, n_covariates])?,
    )?;
    let a = read(a, &cp, "a", Some(v.dtype), 0)?;
    let device = current_device(&cp, &[&v, &cats, &a])?;
    a.require_disjoint(&v)?;
    a.require_disjoint(&cats)?;
    launch(
        device,
        "harmony_scatter",
        v.dtype,
        n_cells * n_pcs,
        stream,
        &mut [
            Arg::Ptr(v.pointer),
            Arg::Ptr(cats.pointer),
            Arg::Ptr(a.pointer),
            Arg::U64(n_cells),
            Arg::U64(n_pcs),
            Arg::U64(n_covariates),
            Arg::U64(a.len.checked_div(n_pcs).unwrap_or(0)),
            Arg::I32(switcher),
        ],
    )
}
#[pyfunction]
#[pyo3(signature=(v,*,cats,n_cells,n_pcs,n_batches,n_covariates=1,switcher,a,n_blocks,stream=0))]
pub fn scatter_add_shared(
    py: Python<'_>,
    v: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    n_cells: u64,
    n_pcs: u64,
    n_batches: u64,
    n_covariates: u64,
    switcher: i32,
    a: &Bound<'_, PyAny>,
    n_blocks: u64,
    stream: usize,
) -> PyResult<()> {
    if n_blocks == 0 {
        return Err(PyValueError::new_err("n_blocks must be positive"));
    }
    let cp = py.import("cupy")?;
    read(a, &cp, "a", None, product(&[n_batches, n_pcs])?)?;
    scatter_add(
        py,
        v,
        cats,
        n_cells,
        n_pcs,
        n_covariates,
        switcher,
        a,
        stream,
    )
}
fn rows_impl(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    idx: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    n_rows: u64,
    n_cols: u64,
    scatter: bool,
    integer: bool,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let src = if integer {
        read(src, &cp, "src", Some(Dtype::I32), 0)?
    } else {
        floating(src, &cp, "src", 0)?
    };
    let dst = read(dst, &cp, "dst", Some(src.dtype), 0)?;
    let idx = read(idx, &cp, "idx", Some(Dtype::I32), n_rows)?;
    let work = product(&[n_rows, n_cols])?;
    if (if scatter { src.len } else { dst.len }) < work {
        return Err(PyValueError::new_err("row buffer is too small"));
    }
    let device = current_device(&cp, &[&src, &idx, &dst])?;
    dst.require_disjoint(&src)?;
    dst.require_disjoint(&idx)?;
    launch(
        device,
        "harmony_rows",
        src.dtype,
        work,
        stream,
        &mut [
            Arg::Ptr(src.pointer),
            Arg::Ptr(idx.pointer),
            Arg::Ptr(dst.pointer),
            Arg::U64(n_rows),
            Arg::U64(n_cols),
            Arg::U64(
                (if scatter { dst.len } else { src.len })
                    .checked_div(n_cols)
                    .unwrap_or(0),
            ),
            Arg::U32(scatter as u32),
        ],
    )
}
#[pyfunction]
#[pyo3(signature=(src,*,idx,dst,n_rows,n_cols,stream=0))]
pub fn gather_rows(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    idx: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    n_rows: u64,
    n_cols: u64,
    stream: usize,
) -> PyResult<()> {
    rows_impl(py, src, idx, dst, n_rows, n_cols, false, false, stream)
}
#[pyfunction]
#[pyo3(signature=(src,*,idx,dst,n_rows,n_cols,stream=0))]
pub fn scatter_rows(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    idx: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    n_rows: u64,
    n_cols: u64,
    stream: usize,
) -> PyResult<()> {
    rows_impl(py, src, idx, dst, n_rows, n_cols, true, false, stream)
}
#[pyfunction]
#[pyo3(signature=(src,*,idx,dst,n,stream=0))]
pub fn gather_int(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    idx: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    n: u64,
    stream: usize,
) -> PyResult<()> {
    rows_impl(py, src, idx, dst, n, 1, false, true, stream)
}
#[pyfunction]
#[pyo3(signature=(E,*,Pr_b,R_sum,n_cats,n_pcs,switcher,stream=0))]
pub fn outer(
    py: Python<'_>,
    E: &Bound<'_, PyAny>,
    Pr_b: &Bound<'_, PyAny>,
    R_sum: &Bound<'_, PyAny>,
    n_cats: u64,
    n_pcs: u64,
    switcher: i32,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let e = floating(E, &cp, "E", product(&[n_cats, n_pcs])?)?;
    let p = read(Pr_b, &cp, "Pr_b", Some(e.dtype), n_cats)?;
    let r = read(R_sum, &cp, "R_sum", Some(e.dtype), n_pcs)?;
    let device = current_device(&cp, &[&e, &p, &r])?;
    e.require_disjoint(&p)?;
    e.require_disjoint(&r)?;
    launch(
        device,
        "harmony_outer",
        e.dtype,
        n_cats * n_pcs,
        stream,
        &mut [
            Arg::Ptr(e.pointer),
            Arg::Ptr(p.pointer),
            Arg::Ptr(r.pointer),
            Arg::U64(n_cats),
            Arg::U64(n_pcs),
            Arg::I32(switcher),
        ],
    )
}
pub(crate) fn colsum_multiprocessors(cp: &Bound<'_, PyModule>, device: usize) -> PyResult<u64> {
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
    };
    static COUNTS: OnceLock<Mutex<HashMap<usize, u64>>> = OnceLock::new();
    let counts = COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(count) = counts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&device)
        .copied()
    {
        return Ok(count);
    }
    let properties = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method1("getDeviceProperties", (device,))?;
    let count = properties
        .get_item("multiProcessorCount")?
        .extract::<u64>()?
        .max(1);
    counts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(device, count);
    Ok(count)
}

fn colsum_impl(
    py: Python<'_>,
    A: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    accumulate: bool,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let a = read(A, &cp, "A", None, product(&[rows, cols])?)?;
    let out = read(out, &cp, "out", Some(a.dtype), cols)?;
    let device = current_device(&cp, &[&a, &out])?;
    out.require_disjoint(&a)?;
    launch_colsum(
        &cp,
        device,
        a.dtype,
        a.pointer,
        out.pointer,
        rows,
        cols,
        accumulate,
        stream,
    )
}

/// Launch a column reduction over buffers already validated by an outer binding.
pub(crate) fn launch_colsum(
    cp: &Bound<'_, PyModule>,
    device: usize,
    dtype: Dtype,
    source: u64,
    destination: u64,
    rows: u64,
    cols: u64,
    accumulate: bool,
    stream: usize,
) -> PyResult<()> {
    if cols == 0 {
        return Ok(());
    }
    let multiprocessors = colsum_multiprocessors(cp, device)?;
    let (blocks, threads, rows_per_tile) = if accumulate {
        let col_tiles = cols.div_ceil(32);
        let target = (multiprocessors * 4 / col_tiles).max(1);
        let rows_per_tile = rows.div_ceil(target).max(32);
        (
            product(&[col_tiles, rows.div_ceil(rows_per_tile)])?,
            256,
            rows_per_tile,
        )
    } else {
        (
            cols.min(multiprocessors * 8),
            rows.div_ceil(32).clamp(1, 32) * 32,
            1,
        )
    };
    if blocks == 0 {
        return Ok(());
    }
    let name = match dtype {
        Dtype::F32 => "harmony_colsum_f32",
        Dtype::F64 => "harmony_colsum_f64",
        Dtype::I32 => "harmony_colsum_i32",
        _ => return Err(PyTypeError::new_err("unsupported column sum dtype")),
    };
    let mut args = [
        Arg::Ptr(source),
        Arg::Ptr(destination),
        Arg::U64(rows),
        Arg::U64(cols),
        Arg::U32(accumulate as u32),
        Arg::U64(rows_per_tile),
    ];
    let mut pointers: Vec<_> = args.iter_mut().map(Arg::pointer).collect();
    // SAFETY: validated extents and exclusive output above; the accumulating
    // entry owns 32 columns per block and exactly eight cooperating warps.
    unsafe {
        runtime::launch(
            device,
            name,
            (blocks.min(65_535) as u32, 1, 1),
            (threads as u32, 1, 1),
            stream,
            &mut pointers,
        )
    }
}
#[pyfunction]
#[pyo3(signature=(A,*,out,rows,cols,stream=0))]
pub fn colsum(
    py: Python<'_>,
    A: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    stream: usize,
) -> PyResult<()> {
    colsum_impl(py, A, out, rows, cols, false, stream)
}
#[pyfunction]
#[pyo3(signature=(A,*,out,rows,cols,stream=0))]
pub fn colsum_atomic(
    py: Python<'_>,
    A: &Bound<'_, PyAny>,
    out: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    stream: usize,
) -> PyResult<()> {
    colsum_impl(py, A, out, rows, cols, true, stream)
}
fn norm_impl(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    l2: bool,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let n = product(&[rows, cols])?;
    let src = floating(src, &cp, "src", n)?;
    let dst = read(dst, &cp, "dst", Some(src.dtype), n)?;
    if src.pointer != dst.pointer {
        dst.require_disjoint(&src)?;
    }
    let device = current_device(&cp, &[&src, &dst])?;
    launch_rows(
        device,
        "harmony_normalize",
        src.dtype,
        rows,
        cols,
        stream,
        &mut [
            Arg::Ptr(src.pointer),
            Arg::Ptr(dst.pointer),
            Arg::U64(rows),
            Arg::U64(cols),
            Arg::U32(l2 as u32),
        ],
    )
}
#[pyfunction]
#[pyo3(signature=(X,*,rows,cols,stream=0))]
pub fn normalize(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    rows: u64,
    cols: u64,
    stream: usize,
) -> PyResult<()> {
    norm_impl(py, X, X, rows, cols, false, stream)
}
#[pyfunction]
#[pyo3(signature=(src,*,dst,n_rows,n_cols,stream=0))]
pub fn l2_row_normalize(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    dst: &Bound<'_, PyAny>,
    n_rows: u64,
    n_cols: u64,
    stream: usize,
) -> PyResult<()> {
    norm_impl(py, src, dst, n_rows, n_cols, true, stream)
}
#[pyfunction]
#[pyo3(signature=(r,*,dot,n,out,stream=0))]
pub fn kmeans_err(
    py: Python<'_>,
    r: &Bound<'_, PyAny>,
    dot: &Bound<'_, PyAny>,
    n: u64,
    out: &Bound<'_, PyAny>,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let r = floating(r, &cp, "r", n)?;
    let dot = read(dot, &cp, "dot", Some(r.dtype), n)?;
    let out = read(out, &cp, "out", Some(r.dtype), 1)?;
    let device = current_device(&cp, &[&r, &dot, &out])?;
    out.require_disjoint(&r)?;
    out.require_disjoint(&dot)?;
    launch(
        device,
        "harmony_kmeans",
        r.dtype,
        n.min(colsum_multiprocessors(&cp, device)? * 8 * 256),
        stream,
        &mut [
            Arg::Ptr(r.pointer),
            Arg::Ptr(dot.pointer),
            Arg::Ptr(out.pointer),
            Arg::U64(n),
        ],
    )
}
#[pyfunction]
#[pyo3(signature=(E,*,O,theta,penalty,n_batches,n_clusters,stabilized,stream=0))]
pub fn penalty(
    py: Python<'_>,
    E: &Bound<'_, PyAny>,
    O: &Bound<'_, PyAny>,
    theta: &Bound<'_, PyAny>,
    penalty: &Bound<'_, PyAny>,
    n_batches: u64,
    n_clusters: u64,
    stabilized: bool,
    stream: usize,
) -> PyResult<()> {
    let cp = py.import("cupy")?;
    let n = product(&[n_batches, n_clusters])?;
    let e = floating(E, &cp, "E", n)?;
    let o = read(O, &cp, "O", Some(e.dtype), n)?;
    let t = read(theta, &cp, "theta", Some(e.dtype), n_batches)?;
    let p = read(penalty, &cp, "penalty", Some(e.dtype), n)?;
    let device = current_device(&cp, &[&e, &o, &t, &p])?;
    for a in [&e, &o, &t] {
        p.require_disjoint(a)?;
    }
    launch(
        device,
        "harmony_penalty",
        e.dtype,
        n,
        stream,
        &mut [
            Arg::Ptr(e.pointer),
            Arg::Ptr(o.pointer),
            Arg::Ptr(t.pointer),
            Arg::Ptr(p.pointer),
            Arg::U64(n_batches),
            Arg::U64(n_clusters),
            Arg::U32(stabilized as u32),
        ],
    )
}
#[pyfunction]
#[pyo3(signature=(similarities,*,penalty,cats,idx_in,R_out,term,n_rows,n_cols,n_covariates=1,stream=0))]
pub fn fused_pen_norm_int(
    py: Python<'_>,
    similarities: &Bound<'_, PyAny>,
    penalty: &Bound<'_, PyAny>,
    cats: &Bound<'_, PyAny>,
    idx_in: &Bound<'_, PyAny>,
    R_out: &Bound<'_, PyAny>,
    term: f64,
    n_rows: u64,
    n_cols: u64,
    n_covariates: u64,
    stream: usize,
) -> PyResult<()> {
    if n_covariates == 0 {
        return Err(PyValueError::new_err(
            "fused_pen_norm requires at least one covariate",
        ));
    }
    let cp = py.import("cupy")?;
    let s = floating(similarities, &cp, "similarities", 0)?;
    let p = read(penalty, &cp, "penalty", Some(s.dtype), 0)?;
    let c = read(
        cats,
        &cp,
        "cats",
        Some(Dtype::I32),
        product(&[n_rows, n_covariates])?,
    )?;
    let idx = read(idx_in, &cp, "idx_in", Some(Dtype::I32), n_rows)?;
    let out = read(
        R_out,
        &cp,
        "R_out",
        Some(s.dtype),
        product(&[n_rows, n_cols])?,
    )?;
    let device = current_device(&cp, &[&s, &p, &c, &idx, &out])?;
    for a in [&s, &p, &c, &idx] {
        out.require_disjoint(a)?;
    }
    launch_rows(
        device,
        pen_kernel(n_covariates),
        s.dtype,
        n_rows,
        n_cols,
        stream,
        &mut [
            Arg::Ptr(s.pointer),
            Arg::Ptr(p.pointer),
            Arg::Ptr(c.pointer),
            Arg::Ptr(idx.pointer),
            Arg::Ptr(out.pointer),
            Arg::U64(n_rows),
            Arg::U64(n_cols),
            Arg::U64(n_covariates),
            Arg::U64(s.len.checked_div(n_cols).unwrap_or(0)),
            Arg::U64(p.len.checked_div(n_cols).unwrap_or(0)),
            scalar(s.dtype, term),
        ],
    )
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! module {($name:literal,$($fun:ident),+)=>{{let m=PyModule::new(parent.py(),concat!("rapids_singlecell._cuda.",$name))?;m.setattr("__backend__","rust")?;$(m.add_function(wrap_pyfunction!($fun,&m)?)?;)+parent.add_submodule(&m)?;}};}
    module!(
        "_harmony_scatter_cuda",
        scatter_add,
        scatter_add_shared,
        gather_rows,
        scatter_rows,
        gather_int
    );
    module!("_harmony_outer_cuda", outer);
    module!("_harmony_colsum_cuda", colsum, colsum_atomic);
    module!("_harmony_kmeans_cuda", kmeans_err);
    module!("_harmony_normalize_cuda", normalize, l2_row_normalize);
    module!("_harmony_pen_cuda", penalty, fused_pen_norm_int);
    Ok(())
}

pub(crate) use crate::blas::gemm;

/// Check writable scratch and outputs before any kernel can mutate them.
pub(crate) fn disjoint(outputs: &[&Array], inputs: &[&Array]) -> PyResult<()> {
    for (i, output) in outputs.iter().enumerate() {
        for input in inputs.iter().chain(outputs[..i].iter()) {
            output.require_disjoint(input)?;
        }
    }
    Ok(())
}
