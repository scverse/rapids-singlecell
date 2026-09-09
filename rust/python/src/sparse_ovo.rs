//! Bounded sparse reference comparisons with compact native segment sorting.
#![allow(clippy::too_many_arguments)]
use crate::{
    array::{Array, Dtype, Layout, current_device},
    rank_support::*,
    sparse_ovr::CscBlock,
    wilcoxon::Stats,
};
use pyo3::{
    exceptions::{PyMemoryError, PyValueError},
    prelude::*,
};

fn output(
    cp: &Bound<'_, PyModule>,
    object: Option<&Bound<'_, PyAny>>,
    count: u64,
) -> PyResult<Option<Array>> {
    object
        .map(|object| {
            let a = read(object, cp, "statistics output", Some(Dtype::F64), Layout::C)?;
            capacity(&a, count, "statistics output")?;
            Ok(a)
        })
        .transpose()
}
fn ptr(a: &Option<Array>) -> Arg {
    Arg::P(a.as_ref().map_or(0, |a| a.pointer))
}

/// Materialize only stored entries in each requested CSC window. Omitted
/// reference and group values contribute analytically to ranks and ties.
pub fn ovo<'py>(
    cp: &Bound<'py, PyModule>,
    rows: usize,
    refs: &Bound<'py, PyAny>,
    grps: &Bound<'py, PyAny>,
    group_offsets: &[usize],
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
    if group_offsets.len() != groups as usize + 1 || groups >= i32::MAX as u64 {
        return Err(PyValueError::new_err(
            "group offsets must match the rank output",
        ));
    }
    let refs = array(cp, refs, "int64", "C")?;
    let grps = array(cp, grps, "int64", "C")?;
    let r = vector(&refs, cp, "reference row IDs", Some(Dtype::I64))?;
    let g = vector(&grps, cp, "group row IDs", Some(Dtype::I64))?;
    if group_offsets.first() != Some(&0)
        || group_offsets.last() != Some(&(g.len as usize))
        || group_offsets.windows(2).any(|w| w[1] < w[0])
    {
        return Err(PyValueError::new_err(
            "group offsets must partition group row IDs",
        ));
    }
    let tie = output(cp, compute.then_some(tie), rank.len)?;
    let stat_count = product(&[groups + 1, columns])?;
    let sums = output(cp, statistics.as_ref().map(|s| s.sums), stat_count)?;
    let nonzero = output(cp, statistics.as_ref().and_then(|s| s.nnz), stat_count)?;
    let mut outputs = vec![&rank];
    outputs.extend(tie.iter());
    outputs.extend(sums.iter());
    outputs.extend(nonzero.iter());
    crate::harmony::disjoint(&outputs, &[&r, &g])?;
    current_device(cp, &[&rank, &r, &g])?;
    for output in &outputs {
        current_device(cp, &[output])?;
    }
    let sizes_host: Vec<u64> = std::iter::once(r.len)
        .chain(group_offsets.windows(2).map(|w| (w[1] - w[0]) as u64))
        .collect();
    let sizes = cp.call_method1("asarray", (&sizes_host, "uint64"))?;
    let sizes_array = vector(&sizes, cp, "population sizes", Some(Dtype::U64))?;
    let offsets = cp.call_method1("asarray", (group_offsets.to_vec(), "uint64"))?;
    let offsets_array = vector(&offsets, cp, "group offsets", Some(Dtype::U64))?;
    let labels = cp.call_method1("full", (rows, -1, "int32"))?;
    let labels_array = vector(&labels, cp, "row memberships", Some(Dtype::I32))?;
    let errors = empty(cp, &[1], "int32", "C", true)?;
    let errors_array = vector(&errors, cp, "selection errors", Some(Dtype::I32))?;
    launch(
        cp,
        "sparse_ovo_memberships",
        r.len + g.len,
        stream,
        &[&r, &g, &offsets_array, &labels_array],
        &mut [
            Arg::P(r.pointer),
            Arg::P(g.pointer),
            Arg::P(offsets_array.pointer),
            Arg::P(labels_array.pointer),
            Arg::P(errors_array.pointer),
            Arg::N(r.len),
            Arg::N(g.len),
            Arg::N(groups),
            Arg::N(rows as u64),
        ],
    )?;
    let error = errors.call_method0("item")?.extract::<i32>()?;
    if error & 1 != 0 {
        return Err(PyValueError::new_err(
            "selected row IDs must be within the source matrix",
        ));
    }
    if error & 2 != 0 {
        return Err(PyValueError::new_err(
            "reference and group row IDs must be unique and disjoint",
        ));
    }
    let memory: (u64, u64) = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("memGetInfo")?
        .extract()?;
    let request = if requested <= 0 {
        64
    } else {
        requested as usize
    };
    let window = request
        .min(columns.max(1) as usize)
        .min((memory.0 / 5 / product(&[rows as u64, 16])?.max(1)).max(1) as usize)
        .max(1);
    for start in (0..columns as usize).step_by(window) {
        let stop = (start + window).min(columns as usize);
        let width = stop - start;
        let block = tile(start, stop)?;
        let data = floating(&block.data, cp, "CSC data", Layout::C)?;
        data.require_vector("CSC data")?;
        let indices = vector(&block.indices, cp, "CSC indices", None)?;
        let index_wide = integer(&indices)?;
        let indptr = vector(&block.indptr, cp, "CSC indptr", None)?;
        let pointer_wide = integer(&indptr)?;
        if data.len != indices.len || indptr.len != width as u64 + 1 {
            return Err(PyValueError::new_err(
                "CSC window has inconsistent array lengths",
            ));
        }
        crate::harmony::disjoint(&outputs, &[&data, &indices, &indptr])?;
        current_device(cp, &[&rank, &data, &indices, &indptr])?;
        if start == 0 {
            for output in &outputs {
                zero(cp, output, stream)?;
            }
        }
        let segments = product(&[groups + 1, width as u64])?;
        let counts = empty(cp, &[segments as usize], "uint64", "C", true)?;
        let count_array = vector(&counts, cp, "stored population counts", Some(Dtype::U64))?;
        let data_wide = wide(data.dtype)?;
        let arguments = |mode, offsets, packed| {
            vec![
                Arg::P(data.pointer),
                Arg::P(indices.pointer),
                Arg::P(indptr.pointer),
                Arg::P(labels_array.pointer),
                Arg::P(count_array.pointer),
                Arg::P(offsets),
                Arg::P(packed),
                ptr(&sums),
                ptr(&nonzero),
                Arg::N(rows as u64),
                Arg::N(width as u64),
                Arg::N(groups),
                Arg::N(data.len),
                Arg::N(columns),
                Arg::N(start as u64),
                Arg::U(data_wide),
                Arg::U(index_wide),
                Arg::U(pointer_wide),
                Arg::U(mode),
            ]
        };
        launch(
            cp,
            "sparse_ovo_csc",
            width as u64 * 256,
            stream,
            &[&data, &indices, &indptr, &count_array],
            &mut arguments(0, 0, 0),
        )?;
        let cumulative = cp.call_method1("cumsum", (&counts,))?;
        let first_zero = empty(cp, &[1], "uint64", "C", true)?;
        let offsets = cp.call_method1("concatenate", ((&first_zero, &cumulative),))?;
        let offsets_array = vector(&offsets, cp, "stored segment offsets", Some(Dtype::U64))?;
        let host_offsets: Vec<u64> = cp
            .call_method1("asnumpy", (&offsets,))?
            .call_method0("tolist")?
            .extract()?;
        for (segment, pair) in host_offsets.windows(2).enumerate() {
            if pair[1] - pair[0] > sizes_host[segment / width] {
                return Err(PyValueError::new_err(
                    "stored segment has more entries than its population",
                ));
            }
        }
        let stored = *host_offsets.last().unwrap();
        let packed = empty(cp, &[stored as usize], "uint32", "C", false)?;
        let packed_array = vector(&packed, cp, "compact keys", Some(Dtype::U32))?;
        zero(cp, &count_array, stream)?;
        launch(
            cp,
            "sparse_ovo_csc",
            width as u64 * 256,
            stream,
            &[&data, &indices, &indptr, &count_array, &packed_array],
            &mut arguments(1, offsets_array.pointer, packed_array.pointer),
        )?;
        let sorted = empty(cp, &[stored as usize], "uint32", "C", false)?;
        let sorted_array = vector(&sorted, cp, "sorted compact keys", Some(Dtype::U32))?;
        let memory: (u64, u64) = cp
            .getattr("cuda")?
            .getattr("runtime")?
            .call_method0("memGetInfo")?
            .extract()?;
        let budget = memory.0 / 5;
        let mut begin = 0usize;
        while begin < segments as usize {
            if host_offsets[begin + 1] == host_offsets[begin] {
                begin += 1;
                continue;
            }
            let mut end = begin;
            let mut longest = 1u64;
            let mut total = 0u64;
            while end < segments as usize {
                let count = host_offsets[end + 1] - host_offsets[end];
                let next_longest = longest.max(count);
                let next_total = total + count;
                let padded = product(&[next_longest, (end - begin + 1) as u64])?;
                let bytes = product(&[padded, 40])? + product(&[(end - begin + 1) as u64, 1024])?;
                if end > begin && (bytes > budget || padded > 4 * next_total.max(1)) {
                    break;
                }
                if bytes > budget {
                    return Err(PyMemoryError::new_err(
                        "insufficient CUDA memory for one sparse ranking segment",
                    ));
                }
                longest = next_longest;
                total = next_total;
                end += 1;
            }
            let nsegments = end - begin;
            let padded = empty(cp, &[nsegments, longest as usize], "uint32", "C", false)?;
            let padded_array = read(
                &padded,
                cp,
                "padded segment keys",
                Some(Dtype::U32),
                Layout::C,
            )?;
            launch(
                cp,
                "sparse_ovo_segment_copy",
                padded_array.len,
                stream,
                &[&packed_array, &offsets_array, &padded_array],
                &mut [
                    Arg::P(packed_array.pointer),
                    Arg::P(offsets_array.pointer),
                    Arg::P(padded_array.pointer),
                    Arg::P(0),
                    Arg::N(begin as u64),
                    Arg::N(nsegments as u64),
                    Arg::N(longest),
                    Arg::U(0),
                ],
            )?;
            let sorted_part = crate::rank_sort::sort(cp, &padded.getattr("T")?, false)?;
            let part = read(
                &sorted_part,
                cp,
                "sorted segment keys",
                Some(Dtype::U32),
                Layout::F,
            )?;
            launch(
                cp,
                "sparse_ovo_segment_copy",
                part.len,
                stream,
                &[&part, &offsets_array, &sorted_array],
                &mut [
                    Arg::P(0),
                    Arg::P(offsets_array.pointer),
                    Arg::P(part.pointer),
                    Arg::P(sorted_array.pointer),
                    Arg::N(begin as u64),
                    Arg::N(nsegments as u64),
                    Arg::N(longest),
                    Arg::U(1),
                ],
            )?;
            begin = end;
        }
        launch(
            cp,
            "sparse_ovo_rank",
            product(&[groups, width as u64, 256])?,
            stream,
            &[&sorted_array, &offsets_array, &sizes_array, &rank],
            &mut [
                Arg::P(sorted_array.pointer),
                Arg::P(offsets_array.pointer),
                Arg::P(sizes_array.pointer),
                Arg::P(rank.pointer),
                ptr(&tie),
                Arg::N(groups),
                Arg::N(width as u64),
                Arg::N(columns),
                Arg::N(start as u64),
            ],
        )?;
    }
    sync(cp)
}
