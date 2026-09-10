//! Adaptive CSR ranking using one immutable sorted-reference cache.
#![allow(clippy::too_many_arguments)]
use super::*;
use crate::{
    host_sparse::{CsrRankPack, Populations},
    staging::BatchStreams,
};
use pyo3::types::PyDict;

pub fn try_rank<'py>(
    cp: &Bound<'py, PyModule>,
    populations: &Populations,
    offsets: &[usize],
    ranks: &Array,
    ties: Option<&Array>,
    sums: Option<&Array>,
    counts: Option<&Array>,
    requested: isize,
    prepare: impl FnOnce() -> PyResult<Option<CsrRankPack>>,
) -> PyResult<bool> {
    let nref = populations.reference_count();
    let ngrp = *offsets.last().unwrap_or(&0);
    let cols = ranks.shape[1];
    // Original medium/large OVO tiers avoid sorting the group populations.
    // Larger populations retain the measured-faster compact sparse backend.
    if nref == 0
        || nref > 4096
        || ngrp == 0
        || cols < 256
        || offsets.len() <= 1
        || offsets.windows(2).any(|span| span[1] - span[0] > 2500)
        || nref.checked_add(ngrp).is_none_or(|rows| rows > 32768)
    {
        return Ok(false);
    }
    let Some(sums) = sums else {
        return Ok(false);
    };
    let (free, _): (u64, u64) = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("memGetInfo")?
        .extract()?;
    let budget = (free / 5).min(64 * 1024 * 1024);
    let mut width = (if requested <= 0 {
        64
    } else {
        requested as usize
    })
    .min(cols)
    .min(65535);
    let mut slots = cols.div_ceil(width).clamp(1, 2);
    // One full input and two sorted-value buffers plus radix tile prefixes;
    // two group windows, pointer/label arrays, and allocator alignment slack.
    let estimate = product(&[nref as u64, cols as u64, 12])?
        .checked_add(product(&[cols as u64, nref.div_ceil(1024) as u64, 256, 8])?)
        .and_then(|size| size.checked_add((ngrp as u64) * width as u64 * slots as u64 * 4))
        .and_then(|size| {
            size.checked_add((nref + ngrp + cols + offsets.len()) as u64 * 16 + 65536)
        });
    if estimate.is_none_or(|size| size > budget) {
        return Ok(false);
    }
    let Some(pack) = prepare()? else {
        return Ok(false);
    };
    if pack.rows != nref + ngrp || pack.cols != cols {
        return Err(PyValueError::new_err("CSR rank pack dimensions changed"));
    }
    if estimate
        .unwrap()
        .checked_add(pack.bytes() as u64 * 2)
        .is_none_or(|size| size > budget)
    {
        return Ok(false);
    }
    // The archived CSR planner expands group slabs beyond the caller's
    // minimum column width when the already reserved memory permits it.
    // Reference sort/cache storage and both upload copies stay in this budget.
    let fixed_bytes =
        estimate.unwrap() - ngrp as u64 * width as u64 * slots as u64 * 4 + pack.bytes() as u64 * 2;
    let available = budget - fixed_bytes;
    let expanded = (available / (ngrp as u64 * slots as u64 * 4))
        .min(cols as u64)
        .min(65535) as usize;
    width = width.max(if expanded < cols {
        expanded / 32 * 32
    } else {
        expanded
    });
    slots = cols.div_ceil(width).clamp(1, 2);
    let analytic = pack.nonnegative_finite && ties.is_some();
    let mut reference_sort;
    let mut initial = BatchStreams::caller(cp)?;
    let (bytes, dtype, nnz) = pack.data();
    let data = initial.upload(0, 0, bytes, dtype, &[nnz], "C")?;
    let columns = initial.upload(
        0,
        1,
        pack.columns(),
        Dtype::U8,
        &[pack.columns().len()],
        "C",
    )?;
    let pointers = initial.upload(0, 2, pack.pointers(), Dtype::U64, &[pack.rows + 1], "C")?;
    let offsets_native: Vec<u64> = offsets.iter().map(|&v| v as u64).collect();
    // SAFETY: initialized primitive native vector, borrowed through upload.
    let offsets_bytes = unsafe {
        std::slice::from_raw_parts(
            offsets_native.as_ptr().cast(),
            std::mem::size_of_val(offsets_native.as_slice()),
        )
    };
    let device_offsets = initial.upload(0, 3, offsets_bytes, Dtype::U64, &[offsets.len()], "C")?;
    let oa = vector(&device_offsets, cp, "CSR group offsets", Some(Dtype::U64))?;
    let data = floating(&data, cp, "packed CSR values", Layout::C)?;
    let columns = vector(&columns, cp, "packed CSR columns", Some(Dtype::U8))?;
    let pointers = vector(&pointers, cp, "packed CSR offsets", Some(Dtype::U64))?;
    capacity(
        &columns,
        product(&[nnz as u64, if pack.narrow() { 2 } else { 4 }])?,
        "packed CSR columns",
    )?;
    capacity(&pointers, (pack.rows + 1) as u64, "packed CSR offsets")?;
    if !pointers.pointer.is_multiple_of(8)
        || !columns
            .pointer
            .is_multiple_of(if pack.narrow() { 2 } else { 4 })
    {
        return Err(PyValueError::new_err("CSR staging storage is misaligned"));
    }
    let narrow = u32::from(pack.narrow());
    let wide = wide(dtype)?;
    let reference_input = initial.allocate(0, &[nref, cols], Dtype::F32, "F")?;
    let reference_input_array = floating(&reference_input, cp, "reference dense input", Layout::F)?;
    reference_sort =
        crate::rank_sort::Workspace::with_allocator(cp, nref, cols, Dtype::F32, None, |bytes| {
            initial.allocate(0, &[bytes], Dtype::U8, "C")
        })?;
    let s = stream(cp)?;
    let extract = |output: &Array, first_row, rows, first, width, s| {
        launch(
            cp,
            "sparse_ovo_csr_dense",
            rows,
            s,
            &[&data, &columns, &pointers, output],
            &mut [
                Arg::P(data.pointer),
                Arg::P(columns.pointer),
                Arg::P(pointers.pointer),
                Arg::P(output.pointer),
                Arg::N(first_row),
                Arg::N(rows),
                Arg::N(first),
                Arg::N(width),
                Arg::N(nnz as u64),
                Arg::U(wide),
                Arg::U(narrow),
            ],
        )
    };
    // All host validation and the conservative memory plan finish before any
    // caller-owned result is changed. Original-precision stats run only once.
    for output in std::iter::once(ranks)
        .chain(ties)
        .chain(Some(sums))
        .chain(counts)
    {
        zero(cp, output, s)?;
    }
    launch(
        cp,
        "sparse_ovo_csr_stats",
        pack.rows as u64 * 256,
        s,
        &[&data, &columns, &pointers, &oa, sums],
        &mut [
            Arg::P(data.pointer),
            Arg::P(columns.pointer),
            Arg::P(pointers.pointer),
            Arg::P(oa.pointer),
            Arg::P(sums.pointer),
            Arg::P(counts.map_or(0, |a| a.pointer)),
            Arg::N(nref as u64),
            Arg::N(pack.rows as u64),
            Arg::N(cols as u64),
            Arg::N((offsets.len() - 1) as u64),
            Arg::N(nnz as u64),
            Arg::U(wide),
            Arg::U(narrow),
        ],
    )?;
    zero(cp, &reference_input_array, s)?;
    extract(&reference_input_array, 0, nref as u64, 0, cols as u64, s)?;
    let reference = reference_sort.sort(cp, &reference_input)?;
    let ra = floating(&reference, cp, "cached reference", Layout::F)?;
    let tie_cache = if ties.is_some() {
        let tie = initial.allocate(0, &[cols], Dtype::F64, "C")?;
        let ta = vector(&tie, cp, "cached reference ties", Some(Dtype::F64))?;
        crate::rank_sort::checked_launch(
            ra.device,
            "rank_ovo_ref_ties",
            cols as u64 * 256,
            s,
            &mut [
                Arg::P(ra.pointer),
                Arg::P(ta.pointer),
                Arg::N(nref as u64),
                Arg::N(cols as u64),
            ],
        )?;
        Some(tie)
    } else {
        None
    };
    initial.finish()?;
    let plan = crate::wilcoxon::OvoTierPlan::new(offsets, ties.is_some());
    let mut group_buffers = vec![None; slots];
    let mut pipeline = if slots > 1 {
        BatchStreams::new(cp, slots)?
    } else {
        BatchStreams::caller(cp)?
    };
    for (batch, first) in (0..cols).step_by(width).enumerate() {
        let slot = batch % slots;
        let _scope = pipeline.enter(slot)?;
        let s = stream(cp)?;
        let stop = (first + width).min(cols);
        let active = stop - first;
        if group_buffers[slot].is_none() {
            group_buffers[slot] =
                Some(pipeline.allocate(slot, &[ngrp * width], Dtype::F32, "C")?);
        }
        let options = PyDict::new(cp.py());
        options.set_item("order", "F")?;
        let group = group_buffers[slot]
            .as_ref()
            .unwrap()
            .get_item(slice(cp.py(), 0, ngrp * active))?
            .call_method("reshape", ((ngrp, active),), Some(&options))?;
        pipeline.retain(slot, group.clone());
        let ga = floating(&group, cp, "CSR group dense window", Layout::F)?;
        zero(cp, &ga, s)?;
        extract(
            &ga,
            nref as u64,
            ngrp as u64,
            first as u64,
            active as u64,
            s,
        )?;
        let reference = reference.get_item((all(cp.py()), slice(cp.py(), first, stop)))?;
        pipeline.retain(slot, reference.clone());
        let ra = floating(&reference, cp, "cached reference window", Layout::F)?;
        let tie = tie_cache
            .as_ref()
            .map(|cache| cache.get_item(slice(cp.py(), first, stop)))
            .transpose()?;
        if let Some(tie) = &tie {
            pipeline.retain(slot, tie.clone());
        }
        let ta = tie
            .as_ref()
            .map(|tie| vector(tie, cp, "cached tie window", Some(Dtype::F64)))
            .transpose()?;
        if analytic {
            let ta = ta.as_ref().expect("analytic ties require a cache");
            let ties = ties.expect("analytic ties require an output");
            launch(
                cp,
                "sparse_ovo_csr_analytic",
                active as u64 * (offsets.len() - 1) as u64 * 256,
                s,
                &[&ra, &ga, &oa, ta, ranks, ties],
                &mut [
                    Arg::P(ra.pointer),
                    Arg::P(ga.pointer),
                    Arg::P(oa.pointer),
                    Arg::P(ta.pointer),
                    Arg::P(ranks.pointer),
                    Arg::P(ties.pointer),
                    Arg::N(nref as u64),
                    Arg::N(ngrp as u64),
                    Arg::N(active as u64),
                    Arg::N((offsets.len() - 1) as u64),
                    Arg::N(cols as u64),
                    Arg::N(first as u64),
                ],
            )?;
        } else {
            crate::wilcoxon::launch_ovo_tiers(
                cp,
                &ra,
                &group,
                &ga,
                &oa,
                &plan,
                ta.as_ref(),
                ranks,
                ties,
                cols,
                first,
                None,
                s,
            )?;
        }
    }
    pipeline.finish()?;
    Ok(true)
}
