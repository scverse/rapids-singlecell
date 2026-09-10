//! Bounded sparse reference comparisons with compact native segment sorting.
#![allow(clippy::too_many_arguments)]
mod dense;
mod nonfinite;
use crate::{
    array::{Array, Dtype, Layout, current_device},
    rank_support::*,
    sparse_ovr::{CscBlock, retain, scratch},
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

struct DeviceMemberships<'py> {
    labels: Bound<'py, PyAny>,
    references: Bound<'py, PyAny>,
    groups: Bound<'py, PyAny>,
    reference_count: u64,
}

/// Device selections require device-side bounds and disjointness validation.
/// The scalar error read completes their temporary uploads before returning.
fn device_memberships<'py>(
    cp: &Bound<'py, PyModule>,
    rows: usize,
    refs: &Bound<'py, PyAny>,
    grps: &Bound<'py, PyAny>,
    group_offsets: &[usize],
    outputs: &[&Array],
) -> PyResult<DeviceMemberships<'py>> {
    let mut completion = crate::staging::BatchStreams::caller(cp)?;
    let refs = array(cp, refs, "int64", "C")?;
    completion.retain(0, refs.clone());
    let grps = array(cp, grps, "int64", "C")?;
    completion.retain(0, grps.clone());
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
    crate::harmony::disjoint(outputs, &[&r, &g])?;
    current_device(cp, &[&r, &g])?;
    let offsets = cp.call_method1("asarray", (group_offsets.to_vec(), "uint64"))?;
    completion.retain(0, offsets.clone());
    let offsets_array = vector(&offsets, cp, "group offsets", Some(Dtype::U64))?;
    let labels = cp.call_method1("full", (rows, -1, "int32"))?;
    completion.retain(0, labels.clone());
    let labels_array = vector(&labels, cp, "row memberships", Some(Dtype::I32))?;
    let errors = empty(cp, &[1], "int32", "C", true)?;
    completion.retain(0, errors.clone());
    let errors_array = vector(&errors, cp, "selection errors", Some(Dtype::I32))?;
    launch(
        cp,
        "sparse_ovo_memberships",
        r.len + g.len,
        stream(cp)?,
        &[&r, &g, &offsets_array, &labels_array, &errors_array],
        &mut [
            Arg::P(r.pointer),
            Arg::P(g.pointer),
            Arg::P(offsets_array.pointer),
            Arg::P(labels_array.pointer),
            Arg::P(errors_array.pointer),
            Arg::N(r.len),
            Arg::N(g.len),
            Arg::N((group_offsets.len() - 1) as u64),
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
    Ok(DeviceMemberships {
        labels,
        references: refs,
        groups: grps,
        reference_count: r.len,
    })
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
    stat_codes: Option<&Bound<'py, PyAny>>,
    host: bool,
    mut dense_pack: impl FnMut(
        &crate::host_sparse::Populations,
    ) -> PyResult<Option<crate::host_sparse::CsrRankPack>>,
    mut tile: impl FnMut(
        usize,
        usize,
        &mut crate::staging::BatchStreams<'py>,
        usize,
    ) -> PyResult<CscBlock<'py>>,
) -> PyResult<()> {
    let caller_stream = stream(cp)?;
    let rank = read(ranks, cp, "rank_sums", Some(Dtype::F64), Layout::C)?;
    let (groups, columns) = matrix(&rank, "rank_sums")?;
    if group_offsets.len() != groups as usize + 1 || groups >= i32::MAX as u64 {
        return Err(PyValueError::new_err(
            "group offsets must match the rank output",
        ));
    }
    let host_populations = if host {
        Some(crate::host_sparse::Populations::new(
            cp.py(),
            refs,
            grps,
            group_offsets,
            rows,
        )?)
    } else {
        None
    };
    let tie = output(cp, compute.then_some(tie), rank.len)?;
    let stat_groups = if stat_codes.is_some() {
        let shape = shape(
            statistics
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("statistics labels require output buffers"))?
                .sums,
        )?;
        if shape.len() != 2 || shape[1] as u64 != columns {
            return Err(PyValueError::new_err(
                "statistics output must match columns",
            ));
        }
        shape[0] as u64
    } else {
        groups + 1
    };
    let stat_count = product(&[stat_groups, columns])?;
    let sums = output(cp, statistics.as_ref().map(|s| s.sums), stat_count)?;
    let nonzero = output(cp, statistics.as_ref().and_then(|s| s.nnz), stat_count)?;
    let mut statistics_upload = stat_codes
        .is_some()
        .then(|| crate::staging::BatchStreams::caller(cp))
        .transpose()?;
    let stat_codes_owner = stat_codes
        .map(|codes| array(cp, codes, "int32", "C"))
        .transpose()?;
    if let (Some(guard), Some(codes)) = (&mut statistics_upload, &stat_codes_owner) {
        guard.retain(0, codes.clone());
    }
    let stat_codes = stat_codes_owner
        .as_ref()
        .map(|codes| vector(codes, cp, "statistics codes", Some(Dtype::I32)))
        .transpose()?;
    if stat_codes
        .as_ref()
        .is_some_and(|codes| codes.len != rows as u64)
    {
        return Err(PyValueError::new_err(
            "statistics codes must match source rows",
        ));
    }
    let mut outputs = vec![&rank];
    outputs.extend(tie.iter());
    outputs.extend(sums.iter());
    outputs.extend(nonzero.iter());
    crate::harmony::disjoint(&outputs, &[])?;
    if let Some(codes) = &stat_codes {
        crate::harmony::disjoint(&outputs, &[codes])?;
        current_device(cp, &[codes])?;
    }
    for output in &outputs {
        current_device(cp, &[output])?;
    }
    // The original host entry points preserve every output when either whole
    // population or the column window is empty. Keep this after validation
    // and before either backend can clear caller-owned results.
    if let Some(populations) = &host_populations
        && (populations.reference_count() == 0
            || group_offsets.last() == Some(&0)
            || groups == 0
            || columns == 0)
    {
        return Ok(());
    }
    if stat_codes.is_none()
        && let Some(populations) = &host_populations
        && dense::try_rank(
            cp,
            populations,
            group_offsets,
            &rank,
            tie.as_ref(),
            sums.as_ref(),
            nonzero.as_ref(),
            requested,
            || dense_pack(populations),
        )?
    {
        return Ok(());
    }
    // Retain the private upload through completion, including an exception
    // before the main batch ring exists. Both packers use these owned labels;
    // caller mutations after native planning cannot change segment capacities.
    let mut membership_upload = host
        .then(|| crate::staging::BatchStreams::caller(cp))
        .transpose()?;
    let (labels, selected_device, reference_count) = if let Some(populations) = &host_populations {
        let labels = populations.labels();
        // SAFETY: i32 has no padding; labels is fully initialized and remains
        // borrowed until upload has copied its bytes into private pinned memory.
        let bytes = unsafe {
            std::slice::from_raw_parts(labels.as_ptr().cast(), std::mem::size_of_val(labels))
        };
        let labels = membership_upload
            .as_mut()
            .expect("host membership upload initialized")
            .upload(0, 0, bytes, Dtype::I32, &[rows], "C")?;
        (labels, None, populations.reference_count() as u64)
    } else {
        let members = device_memberships(cp, rows, refs, grps, group_offsets, &outputs)?;
        (
            members.labels,
            Some((members.references, members.groups)),
            members.reference_count,
        )
    };
    let labels_array = vector(&labels, cp, "row memberships", Some(Dtype::I32))?;
    let sizes_host: Vec<u64> = std::iter::once(reference_count)
        .chain(group_offsets.windows(2).map(|w| (w[1] - w[0]) as u64))
        .collect();
    let sizes = cp.call_method1("asarray", (&sizes_host, "uint64"))?;
    let sizes_array = vector(&sizes, cp, "population sizes", Some(Dtype::U64))?;
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
    for output in &outputs {
        zero(cp, output, caller_stream)?;
    }
    // Compact device segments complete faster on the caller stream; host
    // inputs retain independent slots to overlap CPU packing and uploads.
    let slots = if host {
        (columns as usize).div_ceil(window).clamp(1, 4)
    } else {
        1
    };
    // Split the existing memory allowance across concurrent stream slots.
    let window = window
        .min((memory.0 / 5 / slots as u64 / product(&[rows as u64, 16])?.max(1)).max(1) as usize);
    // Keep each stream's sort owners alive until the staging ring has waited
    // for every pending batch, including when a later validation fails.
    let mut sort_caches: Vec<Option<crate::sparse_ovr::SortScratch<'py>>> =
        (0..slots).map(|_| None).collect();
    let mut nonfinite_rows = None;
    let mut staging = if slots > 1 {
        Some(crate::staging::BatchStreams::new(cp, slots)?)
    } else {
        Some(crate::staging::BatchStreams::caller(cp)?)
    };
    for start in (0..columns as usize).step_by(window) {
        let slot = (start / window) % slots;
        let _scope = staging.as_mut().map(|ring| ring.enter(slot)).transpose()?;
        let stream = crate::rank_support::stream(cp)?;
        let stop = (start + window).min(columns as usize);
        let width = stop - start;
        let block = tile(
            start,
            stop,
            staging.as_mut().expect("completion guard initialized"),
            slot,
        )?;
        let planned_offsets = host_populations
            .as_ref()
            .map(|populations| block.population_offsets(cp.py(), populations))
            .transpose()?
            .flatten();
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
        if data.len != indices.len || indptr.len != width as u64 + 1 {
            return Err(PyValueError::new_err(
                "CSC window has inconsistent array lengths",
            ));
        }
        crate::harmony::disjoint(&outputs, &[&data, &indices, &indptr])?;
        current_device(cp, &[&rank, &data, &indices, &indptr])?;
        let segments = product(&[groups + 1, width as u64])?;
        let counts = scratch(
            cp,
            &mut staging,
            slot,
            host,
            &[segments as usize + 1],
            Dtype::U64,
            true,
        )?;
        let count_array = vector(&counts, cp, "stored population counts", Some(Dtype::U64))?;
        let data_wide = wide(data.dtype)?;
        let arguments = |mode, offsets, packed| {
            vec![
                Arg::P(data.pointer),
                Arg::P(indices.pointer),
                Arg::P(indptr.pointer),
                Arg::P(labels_array.pointer),
                Arg::P(stat_codes.as_ref().map_or(0, |codes| codes.pointer)),
                Arg::P(count_array.pointer),
                Arg::P(offsets),
                Arg::P(packed),
                ptr(&sums),
                ptr(&nonzero),
                Arg::N(rows as u64),
                Arg::N(width as u64),
                Arg::N(groups),
                Arg::N(stat_groups),
                Arg::N(data.len),
                Arg::N(columns),
                Arg::N(start as u64),
                Arg::U(data_wide),
                Arg::U(index_wide),
                Arg::U(pointer_wide),
                Arg::U(mode),
            ]
        };
        let (offsets, host_offsets, has_nan) = if let Some(plan) = planned_offsets {
            let host_offsets = plan.offsets;
            // SAFETY: u64 has no padding, and this immutable view borrows a
            // fully initialized native vector until the pinned copy completes.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    host_offsets.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(host_offsets.as_slice()),
                )
            };
            let offsets = staging
                .as_mut()
                .expect("completion guard initialized")
                .upload(slot, 3, bytes, Dtype::U64, &[host_offsets.len()], "C")?;
            (offsets, host_offsets, plan.has_nan)
        } else {
            launch(
                cp,
                "sparse_ovo_csc",
                width as u64 * 256,
                stream,
                &[&data, &indices, &indptr, &count_array],
                &mut arguments(0, 0, 0),
            )?;
            let cumulative = cp.call_method1("cumsum", (&counts,))?;
            retain(&mut staging, slot, &cumulative);
            let first_zero = empty(cp, &[1], "uint64", "C", true)?;
            retain(&mut staging, slot, &first_zero);
            let offsets = cp.call_method1("concatenate", ((&first_zero, &cumulative),))?;
            retain(&mut staging, slot, &offsets);
            let mut host_offsets: Vec<u64> = cp
                .call_method1("asnumpy", (&offsets,))?
                .call_method0("tolist")?
                .extract()?;
            // The last cumulative entry includes a selected-NaN flag. Read
            // it in the same transfer already required to size compact data.
            let flagged_total = host_offsets
                .pop()
                .ok_or_else(|| PyValueError::new_err("missing sparse population flag"))?;
            let has_nan = flagged_total != *host_offsets.last().unwrap_or(&0);
            (offsets, host_offsets, has_nan)
        };
        let offsets_array = vector(&offsets, cp, "stored segment offsets", Some(Dtype::U64))?;
        for (segment, pair) in host_offsets.windows(2).enumerate() {
            if pair[1] - pair[0] > sizes_host[segment / width] {
                return Err(PyValueError::new_err(
                    "stored segment has more entries than its population",
                ));
            }
        }
        let stored = *host_offsets.last().unwrap();
        let packed = scratch(
            cp,
            &mut staging,
            slot,
            host,
            &[stored as usize],
            Dtype::U32,
            false,
        )?;
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
        if has_nan {
            if nonfinite_rows.is_none() {
                nonfinite_rows = Some(nonfinite::Rows::new(
                    cp,
                    host_populations.as_ref(),
                    selected_device.as_ref(),
                    reference_count as usize + group_offsets.last().copied().unwrap_or(0),
                    group_offsets.len(),
                    memory.0 / 5 / slots as u64,
                )?);
            }
            nonfinite::rank(
                cp,
                nonfinite_rows.as_ref().unwrap(),
                reference_count as usize,
                group_offsets,
                &data,
                &indices,
                &indptr,
                rows,
                width,
                start,
                &rank,
                tie.as_ref(),
                memory.0 / 5 / slots as u64,
                stream,
            )?;
            continue;
        }
        let sorted = scratch(
            cp,
            &mut staging,
            slot,
            host,
            &[stored as usize],
            Dtype::U32,
            false,
        )?;
        let sorted_array = vector(&sorted, cp, "sorted compact keys", Some(Dtype::U32))?;
        let memory: (u64, u64) = cp
            .getattr("cuda")?
            .getattr("runtime")?
            .call_method0("memGetInfo")?
            .extract()?;
        let budget = memory.0 / 5 / slots as u64;
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
            let sorter = crate::sparse_ovr::sort_scratch(
                cp,
                &mut sort_caches[slot],
                longest as usize,
                nsegments,
                false,
                budget,
                &mut staging,
                slot,
                host,
            )?;
            let padded = scratch(
                cp,
                &mut staging,
                slot,
                host,
                &[nsegments, longest as usize],
                Dtype::U32,
                false,
            )?;
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
            let sorted_part = sorter.sort(cp, &padded.getattr("T")?)?;
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
    if let Some(ring) = &staging {
        ring.finish()?;
    }
    sync(cp)
}
