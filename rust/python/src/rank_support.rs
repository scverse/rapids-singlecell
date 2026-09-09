//! Shared validated array and batching operations for ranking bindings.
use crate::{
    array::{Array, Dtype, Layout, current_device},
    runtime,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PySlice},
};
use std::ffi::c_void;

pub enum Arg {
    P(u64),
    N(u64),
    U(u32),
    F(f64),
}
impl Arg {
    fn pointer(&mut self) -> *mut c_void {
        match self {
            Self::P(x) | Self::N(x) => (x as *mut u64).cast(),
            Self::U(x) => (x as *mut u32).cast(),
            Self::F(x) => (x as *mut f64).cast(),
        }
    }
}
pub fn launch(
    cp: &Bound<'_, PyModule>,
    name: &str,
    work: u64,
    stream: usize,
    arrays: &[&Array],
    args: &mut [Arg],
) -> PyResult<()> {
    let device = current_device(cp, arrays)?;
    if work == 0 {
        return Ok(());
    }
    let mut pointers: Vec<_> = args.iter_mut().map(Arg::pointer).collect();
    // SAFETY: callers validate allocation extents, dtype, layout and aliasing.
    unsafe {
        runtime::launch(
            device,
            name,
            (work.div_ceil(256).min(65535) as u32, 1, 1),
            (256, 1, 1),
            stream,
            &mut pointers,
        )
    }
}
pub fn read(
    obj: &Bound<'_, PyAny>,
    cp: &Bound<'_, PyModule>,
    name: &str,
    dtype: Option<Dtype>,
    layout: Layout,
) -> PyResult<Array> {
    Array::read(obj, cp, name, dtype, None, layout)
}
pub fn vector(
    obj: &Bound<'_, PyAny>,
    cp: &Bound<'_, PyModule>,
    name: &str,
    dtype: Option<Dtype>,
) -> PyResult<Array> {
    let a = read(obj, cp, name, dtype, Layout::C)?;
    a.require_vector(name)?;
    Ok(a)
}
pub fn floating(
    obj: &Bound<'_, PyAny>,
    cp: &Bound<'_, PyModule>,
    name: &str,
    layout: Layout,
) -> PyResult<Array> {
    let a = read(obj, cp, name, None, layout)?;
    if !matches!(a.dtype, Dtype::F32 | Dtype::F64) {
        return Err(PyTypeError::new_err(format!(
            "{name} must have dtype float32 or float64"
        )));
    }
    Ok(a)
}
pub fn wide(dtype: Dtype) -> PyResult<u32> {
    match dtype {
        Dtype::F32 | Dtype::I32 => Ok(0),
        Dtype::F64 | Dtype::I64 => Ok(1),
        _ => Err(PyTypeError::new_err(
            "expected float32/float64 or int32/int64",
        )),
    }
}
pub fn integer(a: &Array) -> PyResult<u32> {
    if !matches!(a.dtype, Dtype::I32 | Dtype::I64) {
        return Err(PyTypeError::new_err(
            "sparse indices must have dtype int32 or int64",
        ));
    }
    wide(a.dtype)
}
pub fn matrix(a: &Array, name: &str) -> PyResult<(u64, u64)> {
    if a.shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must be two-dimensional"
        )));
    }
    Ok((a.shape[0] as u64, a.shape[1] as u64))
}
pub fn product(dims: &[u64]) -> PyResult<u64> {
    dims.iter()
        .try_fold(1u64, |a, &b| {
            a.checked_mul(b).filter(|&n| n <= isize::MAX as u64 / 8)
        })
        .ok_or_else(|| PyValueError::new_err("dimensions exceed addressable memory"))
}
pub fn capacity(a: &Array, n: u64, name: &str) -> PyResult<()> {
    if a.len < n {
        return Err(PyValueError::new_err(format!(
            "{name} requires at least {n} elements"
        )));
    }
    Ok(())
}
pub fn zero(cp: &Bound<'_, PyModule>, a: &Array, stream: usize) -> PyResult<()> {
    if a.len > 0 {
        cp.getattr("cuda")?.getattr("runtime")?.call_method1(
            "memsetAsync",
            (a.pointer, 0, a.len * a.dtype.size(), stream),
        )?;
    }
    Ok(())
}
pub fn stream(cp: &Bound<'_, PyModule>) -> PyResult<usize> {
    cp.getattr("cuda")?
        .call_method0("get_current_stream")?
        .getattr("ptr")?
        .extract()
}
pub fn sync(cp: &Bound<'_, PyModule>) -> PyResult<()> {
    cp.getattr("cuda")?
        .call_method0("get_current_stream")?
        .call_method0("synchronize")?;
    Ok(())
}
pub fn shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    obj.getattr("shape")?.extract()
}
pub fn slice(py: Python<'_>, start: usize, stop: usize) -> Bound<'_, PySlice> {
    PySlice::new(py, start as isize, stop as isize, 1)
}
pub fn all(py: Python<'_>) -> Bound<'_, PySlice> {
    PySlice::new(py, 0, isize::MAX, 1)
}
pub fn window<'py>(
    obj: &Bound<'py, PyAny>,
    start: usize,
    stop: usize,
) -> PyResult<Bound<'py, PyAny>> {
    obj.get_item((all(obj.py()), slice(obj.py(), start, stop)))
}
pub fn array<'py>(
    cp: &Bound<'py, PyModule>,
    obj: &Bound<'py, PyAny>,
    dtype: &str,
    order: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let kw = PyDict::new(cp.py());
    kw.set_item("dtype", dtype)?;
    kw.set_item("order", order)?;
    cp.getattr("asarray")?.call((obj,), Some(&kw))
}
pub fn empty<'py>(
    cp: &Bound<'py, PyModule>,
    dims: &[usize],
    dtype: &str,
    order: &str,
    zeros: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let kw = PyDict::new(cp.py());
    kw.set_item("dtype", dtype)?;
    kw.set_item("order", order)?;
    cp.getattr(if zeros { "zeros" } else { "empty" })?
        .call((dims.to_vec(),), Some(&kw))
}
pub fn range(start: isize, stop: isize, columns: usize) -> PyResult<(usize, usize)> {
    let stop = if stop < 0 { columns as isize } else { stop };
    if start < 0 || stop < start || stop as usize > columns {
        return Err(PyValueError::new_err("invalid column range"));
    }
    Ok((start as usize, stop as usize))
}
/// Cap dense staging and sort scratch to 20% of currently available memory.
pub fn batch(
    cp: &Bound<'_, PyModule>,
    rows: usize,
    requested: isize,
    columns: usize,
) -> PyResult<usize> {
    let mem: (u64, u64) = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("memGetInfo")?
        .extract()?;
    // Native radix sorting owns value/index ping-pong buffers plus one 256-bin
    // histogram per 256 rows. Small columns still need a full histogram tile.
    let values = product(&[rows as u64, 48])?;
    let histograms = product(&[(rows as u64).div_ceil(256), 1024])?;
    let per_col = values
        .checked_add(histograms)
        .ok_or_else(|| PyValueError::new_err("ranking scratch size overflow"))?
        .max(2048);
    let available = (mem.0 / 5 / per_col) as usize;
    if columns > 0 && available == 0 {
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "insufficient CUDA memory for one ranking column",
        ));
    }
    Ok((if requested <= 0 {
        64
    } else {
        requested as usize
    })
    .min(available.max(1))
    .min(columns.max(1))
    .max(1))
}

#[allow(clippy::too_many_arguments)]
pub fn stats(
    cp: &Bound<'_, PyModule>,
    x: &Bound<'_, PyAny>,
    codes: &Bound<'_, PyAny>,
    mask: Option<&Bound<'_, PyAny>>,
    sums: Option<&Bound<'_, PyAny>>,
    squares: Option<&Bound<'_, PyAny>>,
    nnz: Option<&Bound<'_, PyAny>>,
    total: Option<&Bound<'_, PyAny>>,
    total_nnz: Option<&Bound<'_, PyAny>>,
    out_cols: u64,
    col_offset: u64,
    stream: usize,
) -> PyResult<()> {
    let x = floating(x, cp, "block", Layout::F)?;
    let (rows, cols) = matrix(&x, "block")?;
    if col_offset
        .checked_add(cols)
        .is_none_or(|end| end > out_cols)
    {
        return Err(PyValueError::new_err(
            "stats output column window exceeds buffer",
        ));
    }
    let c = vector(codes, cp, "group_codes", Some(Dtype::I32))?;
    capacity(&c, rows, "group_codes")?;
    let m = mask
        .map(|o| vector(o, cp, "mask", Some(Dtype::Bool)))
        .transpose()?;
    if let Some(m) = &m {
        capacity(m, rows, "mask")?;
    }
    let outputs = [sums, squares, nnz, total, total_nnz]
        .into_iter()
        .map(|o| {
            o.map(|o| read(o, cp, "stats output", Some(Dtype::F64), Layout::C))
                .transpose()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let groups = outputs
        .iter()
        .take(3)
        .flatten()
        .next()
        .map(|a| a.shape.first().copied().unwrap_or(0) as u64)
        .unwrap_or(0);
    for (i, a) in outputs.iter().enumerate() {
        if let Some(a) = a {
            capacity(
                a,
                product(&[if i < 3 { groups } else { 1 }, out_cols])?,
                "stats output",
            )?;
            a.require_disjoint(&x)?;
            a.require_disjoint(&c)?;
            if let Some(m) = &m {
                a.require_disjoint(m)?;
            }
            for b in outputs.iter().take(i).flatten() {
                a.require_disjoint(b)?;
            }
        }
    }
    let p = |i: usize| outputs[i].as_ref().map_or(0, |a| a.pointer);
    let mut arrays = vec![&x, &c];
    arrays.extend(m.iter());
    arrays.extend(outputs.iter().flatten());
    let c_order = 0;
    // Callers materialize F-order tiles before entering this shared operation.
    launch(
        cp,
        "rank_stats",
        x.len,
        stream,
        &arrays,
        &mut [
            Arg::P(x.pointer),
            Arg::P(c.pointer),
            Arg::P(m.as_ref().map_or(0, |m| m.pointer)),
            Arg::P(p(0)),
            Arg::P(p(1)),
            Arg::P(p(2)),
            Arg::P(p(3)),
            Arg::P(p(4)),
            Arg::N(rows),
            Arg::N(cols),
            Arg::N(groups),
            Arg::N(out_cols),
            Arg::N(col_offset),
            Arg::U(wide(x.dtype)?),
            Arg::U(c_order),
        ],
    )
}
