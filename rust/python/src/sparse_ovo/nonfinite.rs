//! Bounded dense fallback for the floating-search semantics of NaN ranks.
#![allow(clippy::too_many_arguments)]
use super::*;
use crate::{host_sparse::Populations, staging::BatchStreams};
use pyo3::types::PyDict;

/// A sorted lookup preserves each row's original within-population position.
/// It scales with selected rows, rather than the full sparse matrix height.
pub struct Rows {
    ids: Vec<u64>,
    positions: Vec<u64>,
}

impl Rows {
    pub fn new(
        cp: &Bound<'_, PyModule>,
        host: Option<&Populations>,
        device: Option<&(Bound<'_, PyAny>, Bound<'_, PyAny>)>,
        selected_count: usize,
        offsets_count: usize,
        budget: u64,
    ) -> PyResult<Self> {
        // Reject an unaffordable map before a rare device selection snapshot
        // can allocate native vectors proportional to that selection.
        affordable_width(cp, selected_count, offsets_count, 1, budget)?;
        let order: Vec<u64> = if let Some(populations) = host {
            populations
                .row_order()
                .iter()
                .map(|&row| row as u64)
                .collect()
        } else {
            let (refs, grps) = device
                .ok_or_else(|| PyValueError::new_err("missing validated sparse row selections"))?;
            let mut refs: Vec<u64> = cp
                .call_method1("asnumpy", (refs,))?
                .call_method0("tolist")?
                .extract()?;
            let groups: Vec<u64> = cp
                .call_method1("asnumpy", (grps,))?
                .call_method0("tolist")?
                .extract()?;
            refs.extend(groups);
            refs
        };
        if order.len() != selected_count {
            return Err(PyValueError::new_err("sparse population snapshot changed"));
        }
        let mut positions: Vec<u64> = (0..order.len() as u64).collect();
        positions.sort_unstable_by_key(|&position| order[position as usize]);
        let ids = positions
            .iter()
            .map(|&position| order[position as usize])
            .collect();
        // Free the temporary source order before any dense GPU allocations.
        drop(order);
        Ok(Self { ids, positions })
    }
}

fn affordable_width(
    cp: &Bound<'_, PyModule>,
    rows: usize,
    offsets: usize,
    cols: usize,
    budget: u64,
) -> PyResult<usize> {
    let (free, _): (u64, u64) = cp
        .getattr("cuda")?
        .getattr("runtime")?
        .call_method0("memGetInfo")?
        .extract()?;
    let budget = budget.min(free / 5);
    let fixed = product(&[rows as u64, 16])?
        .checked_add(product(&[offsets as u64, 32])?)
        .and_then(|bytes| bytes.checked_add(65536))
        .ok_or_else(|| PyMemoryError::new_err("sparse NaN workspace size overflow"))?;
    // Includes input windows, both radix value buffers, tile histograms and
    // local prefixes, reference ties, and bounded allocator alignment slack.
    // Huge-segment metadata is charged separately above. This estimate is
    // deliberately conservative for small-sort and medium/large OVO tiers.
    let per_col = product(&[rows as u64, 64])?
        .checked_add(product(&[offsets as u64, 64])?)
        .and_then(|bytes| bytes.checked_add(16384))
        .ok_or_else(|| PyMemoryError::new_err("sparse NaN workspace size overflow"))?;
    let available = budget
        .checked_sub(fixed)
        .ok_or_else(|| PyMemoryError::new_err("insufficient CUDA memory for sparse NaN ranking"))?;
    let width = cols.min(65535).min((available / per_col) as usize);
    if width == 0 {
        return Err(PyMemoryError::new_err(
            "insufficient CUDA memory for one sparse NaN ranking column",
        ));
    }
    Ok(width)
}

fn upload<'py>(
    ring: &mut BatchStreams<'py>,
    field: usize,
    data: &[u64],
) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: initialized primitive values have no padding; upload finishes
    // copying these borrowed native bytes before it returns.
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) };
    ring.upload(0, field, bytes, Dtype::U64, &[data.len()], "C")
}

fn window<'py>(
    cp: &Bound<'py, PyModule>,
    storage: &Bound<'py, PyAny>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let options = PyDict::new(cp.py());
    options.set_item("order", "F")?;
    storage
        .get_item(slice(cp.py(), 0, rows * cols))?
        .call_method("reshape", ((rows, cols),), Some(&options))
}

pub fn rank<'py>(
    cp: &Bound<'py, PyModule>,
    selected: &Rows,
    nref: usize,
    offsets: &[usize],
    data: &Array,
    indices: &Array,
    pointers: &Array,
    source_rows: usize,
    source_cols: usize,
    output_start: usize,
    ranks: &Array,
    ties: Option<&Array>,
    budget: u64,
    stream: usize,
) -> PyResult<()> {
    let ngrp = offsets.last().copied().unwrap_or(0);
    let rows = nref
        .checked_add(ngrp)
        .ok_or_else(|| PyValueError::new_err("sparse population size overflow"))?;
    if selected.ids.len() != rows || selected.positions.len() != rows {
        return Err(PyValueError::new_err("sparse population snapshot changed"));
    }
    let width = affordable_width(cp, rows, offsets.len(), source_cols, budget)?;
    let plan = crate::wilcoxon::OvoTierPlan::new(offsets, ties.is_some());
    // Scratch owners precede the completion guard so every failure waits for
    // pending kernels before their arenas can be returned to the allocator.
    let mut reference_sort;
    let mut group_sort;
    let mut completion = BatchStreams::caller(cp)?;
    let ids = upload(&mut completion, 0, &selected.ids)?;
    let positions = upload(&mut completion, 1, &selected.positions)?;
    let offsets_native: Vec<u64> = offsets.iter().map(|&v| v as u64).collect();
    let offsets_device = upload(&mut completion, 2, &offsets_native)?;
    let ids_array = vector(&ids, cp, "selected row IDs", Some(Dtype::U64))?;
    let positions_array = vector(&positions, cp, "selected row positions", Some(Dtype::U64))?;
    let offsets_array = vector(&offsets_device, cp, "NaN group offsets", Some(Dtype::U64))?;
    let reference_input = completion.allocate(0, &[nref * width], Dtype::F32, "C")?;
    let group_input = completion.allocate(0, &[ngrp * width], Dtype::F32, "C")?;
    reference_sort =
        crate::rank_sort::Workspace::with_allocator(cp, nref, width, Dtype::F32, None, |bytes| {
            completion.allocate(0, &[bytes], Dtype::U8, "C")
        })?;
    group_sort = if plan.huge_groups.is_empty() {
        None
    } else {
        Some(crate::rank_sort::Segmented::new(
            cp,
            offsets,
            &plan.huge_groups,
            width,
        )?)
    };
    let reference_ties = if ties.is_some() {
        let owner = completion.allocate(0, &[width], Dtype::F64, "C")?;
        Some(vector(&owner, cp, "NaN reference ties", Some(Dtype::F64))?)
    } else {
        None
    };
    for first in (0..source_cols).step_by(width) {
        let active = width.min(source_cols - first);
        let reference = window(cp, &reference_input, nref, active)?;
        completion.retain(0, reference.clone());
        let group = window(cp, &group_input, ngrp, active)?;
        completion.retain(0, group.clone());
        let ra = floating(&reference, cp, "NaN reference input", Layout::F)?;
        let ga = floating(&group, cp, "NaN group input", Layout::F)?;
        reference.call_method1("fill", (0,))?;
        group.call_method1("fill", (0,))?;
        launch(
            cp,
            "sparse_ovo_nan_dense",
            active as u64 * 256,
            stream,
            &[
                data,
                indices,
                pointers,
                &ids_array,
                &positions_array,
                &ra,
                &ga,
            ],
            &mut [
                Arg::P(data.pointer),
                Arg::P(indices.pointer),
                Arg::P(pointers.pointer),
                Arg::P(ids_array.pointer),
                Arg::P(positions_array.pointer),
                Arg::P(ra.pointer),
                Arg::P(ga.pointer),
                Arg::N(source_rows as u64),
                Arg::N(source_cols as u64),
                Arg::N(data.len),
                Arg::N(rows as u64),
                Arg::N(nref as u64),
                Arg::N(first as u64),
                Arg::N(active as u64),
                Arg::U(wide(data.dtype)?),
                Arg::U(integer(indices)?),
                Arg::U(integer(pointers)?),
            ],
        )?;
        let reference = reference_sort.sort(cp, &reference)?;
        completion.retain(0, reference.clone());
        let ra = floating(&reference, cp, "NaN sorted reference", Layout::F)?;
        if let Some(base) = &reference_ties {
            crate::rank_sort::checked_launch(
                ra.device,
                "rank_ovo_ref_ties",
                active as u64 * 256,
                stream,
                &mut [
                    Arg::P(ra.pointer),
                    Arg::P(base.pointer),
                    Arg::N(nref as u64),
                    Arg::N(active as u64),
                ],
            )?;
        }
        crate::wilcoxon::launch_ovo_tiers(
            cp,
            &ra,
            &group,
            &ga,
            &offsets_array,
            &plan,
            reference_ties.as_ref(),
            ranks,
            ties,
            ranks.shape[1],
            output_start + first,
            group_sort.as_mut(),
            stream,
        )?;
    }
    completion.finish()
}
