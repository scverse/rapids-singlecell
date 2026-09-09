//! Compact CSC sorting for sparse one-versus-rest Wilcoxon ranks.
#![allow(clippy::too_many_arguments)]
use crate::{
    array::{Array, Dtype, Layout, current_device},
    rank_support::*,
    wilcoxon::Stats,
};
use pyo3::{
    exceptions::{PyMemoryError, PyValueError},
    prelude::*,
};

/// A single bounded CSC window, retaining the original values for statistics.
pub struct CscBlock<'py> {
    pub data: Bound<'py, PyAny>,
    pub indices: Bound<'py, PyAny>,
    pub indptr: Bound<'py, PyAny>,
}

fn optional_output(
    cp: &Bound<'_, PyModule>,
    object: Option<&Bound<'_, PyAny>>,
    count: u64,
) -> PyResult<Option<Array>> {
    object
        .map(|object| {
            let output = read(object, cp, "statistics output", Some(Dtype::F64), Layout::C)?;
            capacity(&output, count, "statistics output")?;
            Ok(output)
        })
        .transpose()
}

fn ptr(array: &Option<Array>) -> Arg {
    Arg::P(array.as_ref().map_or(0, |array| array.pointer))
}

/// Sort only stored CSC values; omitted zeros receive their exact average rank.
/// The callback materializes at most the requested column window. Within that
/// window, highly uneven columns are split again to cap padding and sort memory.
pub fn ovr<'py>(
    cp: &Bound<'py, PyModule>,
    rows: usize,
    codes: &Bound<'py, PyAny>,
    sizes: &Bound<'py, PyAny>,
    ranks: &Bound<'py, PyAny>,
    tie: &Bound<'py, PyAny>,
    compute: bool,
    requested: isize,
    statistics: Option<Stats<'_, 'py>>,
    mut tile: impl FnMut(usize, usize) -> PyResult<CscBlock<'py>>,
) -> PyResult<()> {
    let stream = stream(cp)?;
    let rank = read(ranks, cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (groups, columns) = matrix(&rank, "rank_sums")?;
    let codes = vector(codes, cp, "group_codes", Some(Dtype::I32))?;
    let sizes = vector(sizes, cp, "group_sizes", Some(Dtype::F64))?;
    if codes.len != rows as u64 || sizes.len != groups {
        return Err(PyValueError::new_err(
            "group metadata has incompatible dimensions",
        ));
    }
    let tie = optional_output(cp, compute.then_some(tie), columns)?;
    let sums = optional_output(cp, statistics.as_ref().map(|s| s.sums), rank.len)?;
    let counts = optional_output(cp, statistics.as_ref().and_then(|s| s.nnz), rank.len)?;
    let totals = optional_output(cp, statistics.as_ref().and_then(|s| s.total), columns)?;
    let total_counts = optional_output(cp, statistics.as_ref().and_then(|s| s.total_nnz), columns)?;
    let mut outputs = vec![&rank];
    outputs.extend(tie.iter());
    outputs.extend(sums.iter());
    outputs.extend(counts.iter());
    outputs.extend(totals.iter());
    outputs.extend(total_counts.iter());
    crate::harmony::disjoint(&outputs, &[&codes, &sizes])?;
    current_device(cp, &[&codes, &sizes, &rank])?;
    for output in &outputs {
        current_device(cp, &[output])?;
    }
    for output in &outputs {
        zero(cp, output, stream)?;
    }
    let cols = columns as usize;
    let requested = if requested <= 0 {
        64
    } else {
        requested as usize
    };
    // The callback may stage a full sparse column. Limit its worst case before
    // transferring host inputs, then budget padded sorting from actual counts.
    let memory: (u64, u64) = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("memGetInfo")?
        .extract()?;
    let bytes_per_full_column = product(&[rows as u64, 16])?.max(1);
    let window = requested
        .min(cols.max(1))
        .min((memory.0 / 5 / bytes_per_full_column).max(1) as usize)
        .max(1);
    for start in (0..cols).step_by(window) {
        let stop = (start + window).min(cols);
        let block = tile(start, stop)?;
        let data = floating(&block.data, cp, "CSC data", Layout::C)?;
        data.require_vector("CSC data")?;
        let indices = vector(&block.indices, cp, "CSC indices", None)?;
        let index_wide = integer(&indices)?;
        let indptr = vector(&block.indptr, cp, "CSC indptr", None)?;
        let pointer_wide = integer(&indptr)?;
        if data.len != indices.len || indptr.len != (stop - start + 1) as u64 {
            return Err(PyValueError::new_err(
                "CSC window has inconsistent array lengths",
            ));
        }
        crate::harmony::disjoint(&outputs, &[&data, &indices, &indptr])?;
        current_device(cp, &[&data, &indices, &indptr, &rank])?;
        // Only the small pointer vector crosses to the host for a memory plan;
        // values and row indices remain on the device throughout ranking.
        let offsets: Vec<u64> = cp
            .call_method1("asnumpy", (&block.indptr,))?
            .call_method0("tolist")?
            .extract()?;
        if offsets
            .windows(2)
            .any(|w| w[1] < w[0] || w[1] - w[0] > rows as u64)
            || offsets.last().is_none_or(|&last| last > data.len)
        {
            return Err(PyValueError::new_err(
                "CSC offsets must describe unique rows within each column",
            ));
        }
        let memory: (u64, u64) = cp
            .getattr("cuda")?
            .getattr("runtime")?
            .call_method0("memGetInfo")?
            .extract()?;
        // Keys, permutation, sort scratch, and CuPy's sort copy are bounded by
        // a conservative 40 bytes per padded entry and 20% of free GPU memory.
        let budget = memory.0 / 5;
        let mut first = 0;
        while first < stop - start {
            let mut end = first;
            let mut longest = 0;
            let mut stored = 0;
            while end < stop - start {
                let count = offsets[end + 1] - offsets[end];
                let next_longest = longest.max(count).max(1);
                let next_stored = stored + count;
                let padded = product(&[next_longest, (end - first + 1) as u64])?;
                let bytes = product(&[padded, 40])? + product(&[(end - first + 1) as u64, 1024])?;
                if end > first
                    && (bytes > budget || padded > 4 * (next_stored + (end - first + 1) as u64))
                {
                    break;
                }
                if bytes > budget {
                    return Err(PyMemoryError::new_err(
                        "insufficient CUDA memory for one sparse ranking column",
                    ));
                }
                longest = next_longest;
                stored = next_stored;
                end += 1;
            }
            let width = end - first;
            let padded = product(&[width as u64, longest])?;
            let keys = empty(cp, &[width, longest as usize], "uint32", "C", false)?;
            let keys_array = read(&keys, cp, "sort keys", Some(Dtype::U32), Layout::C)?;
            launch(
                cp,
                "sparse_ovr_pack",
                padded,
                stream,
                &[&data, &indices, &indptr, &keys_array],
                &mut [
                    Arg::P(data.pointer),
                    Arg::P(indices.pointer),
                    Arg::P(indptr.pointer),
                    Arg::P(codes.pointer),
                    Arg::P(keys_array.pointer),
                    ptr(&sums),
                    ptr(&counts),
                    ptr(&totals),
                    ptr(&total_counts),
                    Arg::N(rows as u64),
                    Arg::N(width as u64),
                    Arg::N(longest),
                    Arg::N(data.len),
                    Arg::N(first as u64),
                    Arg::N((start + first) as u64),
                    Arg::N(columns),
                    Arg::N(groups),
                    Arg::U(wide(data.dtype)?),
                    Arg::U(index_wide),
                    Arg::U(pointer_wide),
                ],
            )?;
            let order = crate::rank_sort::sort(cp, &keys.getattr("T")?, true)?.getattr("T")?;
            let order = read(&order, cp, "sorted positions", Some(Dtype::I64), Layout::C)?;
            let zeros = empty(cp, &[width], "float64", "C", false)?;
            let zeros = vector(&zeros, cp, "zero ranks", Some(Dtype::F64))?;
            let bounds = empty(cp, &[2 * width], "uint64", "C", false)?;
            let bounds = vector(&bounds, cp, "zero bounds", Some(Dtype::U64))?;
            launch(
                cp,
                "sparse_ovr_base",
                width as u64 * groups.max(1),
                stream,
                &[&keys_array, &order, &rank, &zeros],
                &mut [
                    Arg::P(keys_array.pointer),
                    Arg::P(order.pointer),
                    Arg::P(indptr.pointer),
                    Arg::P(sizes.pointer),
                    Arg::P(rank.pointer),
                    ptr(&tie),
                    Arg::P(zeros.pointer),
                    Arg::P(bounds.pointer),
                    Arg::N(rows as u64),
                    Arg::N(width as u64),
                    Arg::N(longest),
                    Arg::N(data.len),
                    Arg::N(first as u64),
                    Arg::N((start + first) as u64),
                    Arg::N(columns),
                    Arg::N(groups),
                    Arg::U(pointer_wide),
                ],
            )?;
            launch(
                cp,
                "sparse_ovr_delta",
                padded,
                stream,
                &[&keys_array, &order, &rank, &zeros],
                &mut [
                    Arg::P(keys_array.pointer),
                    Arg::P(order.pointer),
                    Arg::P(indices.pointer),
                    Arg::P(indptr.pointer),
                    Arg::P(codes.pointer),
                    Arg::P(zeros.pointer),
                    Arg::P(bounds.pointer),
                    Arg::P(rank.pointer),
                    ptr(&tie),
                    Arg::N(rows as u64),
                    Arg::N(width as u64),
                    Arg::N(longest),
                    Arg::N(data.len),
                    Arg::N(first as u64),
                    Arg::N((start + first) as u64),
                    Arg::N(columns),
                    Arg::N(groups),
                    Arg::U(index_wide),
                    Arg::U(pointer_wide),
                ],
            )?;
            first = end;
        }
    }
    if let Some(tie) = &tie {
        launch(
            cp,
            "rank_tie_finish",
            columns,
            stream,
            &[tie],
            &mut [Arg::P(tie.pointer), Arg::N(rows as u64), Arg::N(columns)],
        )?;
    }
    sync(cp)
}
