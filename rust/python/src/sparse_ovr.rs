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
pub struct CscArrays<'py> {
    pub data: Bound<'py, PyAny>,
    pub indices: Bound<'py, PyAny>,
    pub indptr: Bound<'py, PyAny>,
}

pub enum CscBlock<'py> {
    Python(CscArrays<'py>),
    Owned(crate::host_sparse::CscWindow),
    Prepared {
        arrays: CscArrays<'py>,
        offsets: Vec<u64>,
    },
}

/// A failed later allocation must not free inputs of an already queued kernel.
pub(crate) fn retain<'py>(
    staging: &mut Option<crate::staging::BatchStreams<'py>>,
    slot: usize,
    object: &Bound<'py, PyAny>,
) {
    if let Some(ring) = staging {
        ring.retain(slot, object.clone());
    }
}

/// Host batches allocate through the persistent caller-stream pool. Retain
/// every owner before initialization so allocation callbacks and later errors
/// cannot release memory still used by asynchronous work.
pub(crate) fn scratch<'py>(
    cp: &Bound<'py, PyModule>,
    staging: &mut Option<crate::staging::BatchStreams<'py>>,
    slot: usize,
    host: bool,
    dims: &[usize],
    dtype: Dtype,
    clear: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let output = if host {
        staging
            .as_mut()
            .expect("completion guard initialized")
            .allocate(slot, dims, dtype, "C")?
    } else {
        let output = empty(cp, dims, dtype.name(), "C", false)?;
        retain(staging, slot, &output);
        output
    };
    if clear {
        output.call_method1("fill", (0,))?;
    }
    Ok(output)
}

/// A stream-local radix workspace reused across compact sparse subwindows.
pub(crate) struct SortScratch<'py> {
    rows: usize,
    cols: usize,
    workspace: crate::rank_sort::Workspace<'py>,
}

pub(crate) fn sort_scratch<'a, 'py>(
    cp: &Bound<'py, PyModule>,
    cache: &'a mut Option<SortScratch<'py>>,
    rows: usize,
    cols: usize,
    indices: bool,
    budget: u64,
    staging: &mut Option<crate::staging::BatchStreams<'py>>,
    slot: usize,
    host: bool,
) -> PyResult<&'a mut crate::rank_sort::Workspace<'py>> {
    if cache
        .as_ref()
        .is_none_or(|scratch| rows > scratch.rows || cols > scratch.cols)
    {
        let mut capacity_rows = rows;
        let mut capacity_cols = cols;
        if let Some(previous) = cache.as_ref() {
            let grow_rows = previous.rows.max(rows);
            let grow_cols = previous.cols.max(cols);
            let estimate = (grow_rows as u64)
                .checked_mul(grow_cols as u64)
                .and_then(|items| items.checked_mul(40))
                .and_then(|bytes| bytes.checked_add(grow_cols as u64 * 1024));
            let requested_items = (rows as u64).saturating_mul(cols as u64);
            let grow_items = (grow_rows as u64).saturating_mul(grow_cols as u64);
            if estimate.is_some_and(|bytes| bytes <= budget)
                && grow_items <= requested_items.saturating_mul(4)
            {
                capacity_rows = grow_rows;
                capacity_cols = grow_cols;
            }
            // Earlier subwindows on this stream may still read the old sort
            // result. Complete those consumers before replacing their owner.
            sync(cp)?;
        }
        *cache = None;
        let mut arena = None;
        let workspace = if host {
            crate::rank_sort::Workspace::with_allocator(
                cp,
                capacity_rows,
                capacity_cols,
                Dtype::U32,
                indices.then_some(Dtype::U32),
                |bytes| {
                    let allocation = staging
                        .as_mut()
                        .expect("completion guard initialized")
                        .allocate(slot, &[bytes], Dtype::U8, "C")?;
                    arena = Some(allocation.clone());
                    Ok(allocation)
                },
            )?
        } else if indices {
            crate::rank_sort::Workspace::with_compact_indices(
                cp,
                capacity_rows,
                capacity_cols,
                Dtype::U32,
            )?
        } else {
            crate::rank_sort::Workspace::new(cp, capacity_rows, capacity_cols, Dtype::U32, false)?
        };
        *cache = Some(SortScratch {
            rows: capacity_rows,
            cols: capacity_cols,
            workspace,
        });
        if let Some(arena) = arena {
            // The cache is declared before the completion guard, and growth
            // synchronizes before replacing it. Transfer its ownership only
            // after construction succeeds; a failed constructor remains
            // covered by the guard's allocation retention.
            staging
                .as_mut()
                .expect("completion guard initialized")
                .transfer_allocation(slot, &arena);
        }
    }
    Ok(&mut cache
        .as_mut()
        .expect("sort workspace initialized")
        .workspace)
}

impl<'py> CscBlock<'py> {
    pub fn population_offsets(
        &self,
        py: Python<'_>,
        populations: &crate::host_sparse::Populations,
    ) -> PyResult<Option<crate::host_sparse::PopulationPlan>> {
        match self {
            Self::Owned(window) => populations.plan(py, window).map(Some),
            _ => Ok(None),
        }
    }

    pub fn arrays(
        data: Bound<'py, PyAny>,
        indices: Bound<'py, PyAny>,
        indptr: Bound<'py, PyAny>,
    ) -> Self {
        Self::Python(CscArrays {
            data,
            indices,
            indptr,
        })
    }

    fn planned_offsets(&self, host: bool) -> PyResult<Option<Vec<u64>>> {
        match self {
            Self::Python(arrays) if host => {
                Ok(Some(arrays.indptr.call_method0("tolist")?.extract()?))
            }
            Self::Python(_) => Ok(None),
            Self::Owned(window) => Ok(Some(window.offsets().to_vec())),
            Self::Prepared { offsets, .. } => Ok(Some(offsets.clone())),
        }
    }

    pub fn into_device(
        self,
        staging: &mut crate::staging::BatchStreams<'py>,
        slot: usize,
        host: bool,
    ) -> PyResult<CscArrays<'py>> {
        match self {
            Self::Prepared { arrays, .. } => Ok(arrays),
            Self::Python(arrays) if !host => Ok(arrays),
            Self::Python(arrays) => Ok(CscArrays {
                data: staging.upload_vector(slot, 0, &arrays.data)?,
                indices: staging.upload_vector(slot, 1, &arrays.indices)?,
                indptr: staging.upload_vector(slot, 2, &arrays.indptr)?,
            }),
            Self::Owned(window) => {
                let upload =
                    |staging: &mut crate::staging::BatchStreams<'py>,
                     field,
                     (bytes, dtype, len): (&[u8], Dtype, usize)| {
                        staging.upload(slot, field, bytes, dtype, &[len], "C")
                    };
                Ok(CscArrays {
                    data: upload(staging, 0, window.data())?,
                    indices: upload(staging, 1, window.indices())?,
                    indptr: upload(staging, 2, window.indptr())?,
                })
            }
        }
    }
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
    host: bool,
    mut tile: impl FnMut(
        usize,
        usize,
        &mut crate::staging::BatchStreams<'py>,
        usize,
    ) -> PyResult<CscBlock<'py>>,
) -> PyResult<()> {
    let caller_stream = stream(cp)?;
    let stream = caller_stream;
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
    // Host packing overlaps other slots' kernels. Device inputs use the
    // caller stream: the compact sorter leaves little independent work to
    // overlap, and extra streams increase measured scheduling costs.
    let slots = if host {
        cols.div_ceil(window).clamp(1, 4)
    } else {
        1
    };
    // Split the existing memory allowance across concurrent stream slots.
    let window = window
        .min((memory.0 / 5 / slots as u64 / product(&[rows as u64, 16])?.max(1)).max(1) as usize);
    // Drop the staging ring (and wait its streams) before freeing sort buffers
    // on all return paths, including exceptions while processing a later batch.
    let mut sort_caches: Vec<Option<SortScratch<'py>>> = (0..slots).map(|_| None).collect();
    let mut staging = if slots > 1 {
        Some(crate::staging::BatchStreams::new(cp, slots)?)
    } else {
        Some(crate::staging::BatchStreams::caller(cp)?)
    };
    for start in (0..cols).step_by(window) {
        let slot = (start / window) % slots;
        let _scope = staging.as_mut().map(|ring| ring.enter(slot)).transpose()?;
        let stream = crate::rank_support::stream(cp)?;
        let stop = (start + window).min(cols);
        let block = tile(
            start,
            stop,
            staging.as_mut().expect("completion guard initialized"),
            slot,
        )?;
        let host_offsets = block.planned_offsets(host)?;
        let block = block.into_device(
            staging.as_mut().expect("completion guard initialized"),
            slot,
            host,
        )?;
        for input in [&block.data, &block.indices, &block.indptr] {
            retain(&mut staging, slot, input);
        }
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
        let offsets: Vec<u64> = if let Some(offsets) = host_offsets {
            offsets
        } else {
            cp.call_method1("asnumpy", (&block.indptr,))?
                .call_method0("tolist")?
                .extract()?
        };
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
        let budget = memory.0 / 5 / slots as u64;
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
            let sorter = sort_scratch(
                cp,
                &mut sort_caches[slot],
                longest as usize,
                width,
                true,
                budget,
                &mut staging,
                slot,
                host,
            )?;
            let keys = scratch(
                cp,
                &mut staging,
                slot,
                host,
                &[width, longest as usize],
                Dtype::U32,
                false,
            )?;
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
            let order = sorter.sort(cp, &keys.getattr("T")?)?.getattr("T")?;
            let order = read(&order, cp, "sorted positions", Some(Dtype::U32), Layout::C)?;
            let shared_bytes = product(&[2 * groups + 32, 8])?;
            if shared_bytes <= 48 * 1024 && rows <= i32::MAX as usize {
                let device = current_device(cp, &[&keys_array, &order, &indices, &indptr, &rank])?;
                crate::rank_sort::checked_launch_shared(
                    device,
                    "sparse_ovr_fused",
                    width as u64 * 256,
                    256,
                    shared_bytes as u32,
                    stream,
                    &mut [
                        Arg::P(keys_array.pointer),
                        Arg::P(order.pointer),
                        Arg::P(indices.pointer),
                        Arg::P(indptr.pointer),
                        Arg::P(codes.pointer),
                        Arg::P(sizes.pointer),
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
                continue;
            }
            // Large group populations retain the bounded global accumulator
            // fallback, avoiding a shared-memory limit on perturbation DE.
            let zeros = scratch(cp, &mut staging, slot, host, &[width], Dtype::F64, false)?;
            let zeros = vector(&zeros, cp, "zero ranks", Some(Dtype::F64))?;
            let bounds = scratch(
                cp,
                &mut staging,
                slot,
                host,
                &[2 * width],
                Dtype::U64,
                false,
            )?;
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
    if let Some(ring) = &staging {
        ring.finish()?;
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
